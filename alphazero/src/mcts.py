import time
import ctypes
import torch
import logging
from enum import IntEnum
from wrapper import (
    Board, Moves,
    init_all_attack_tables, parse_fen,
    generate_moves, make_move, copy_board,
    get_move_source, get_move_target, get_move_promoted,
    get_bit, is_in_check
)
from config import Config
from utils import move_to_str


class bcolors:
    HEADER = '[95m'
    OKBLUE = '[94m'
    OKCYAN = '[96m'
    OKGREEN = '[92m'
    WARNING = '[93m'
    FAIL = '[91m'
    ENDC = '[0m'
    BOLD = '[1m'
    UNDERLINE = '[4m'

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s | {bcolors.OKCYAN}%(levelname)s{bcolors.ENDC} | %(message)s',
    datefmt="%H:%M:%S",
)

class Pieces(IntEnum):
    P=0
    N=1
    B=2
    R=3
    Q=4
    K=5
    p=6
    n=7
    b=8
    r=9
    q=10
    k=11

directions = {
    (-1, 0):  0, # N
    (-1, 1):  1, # NE
    (0, 1):   2, # E
    (1, 1):   3, # SE
    (1, 0):   4, # S
    (1, -1):  5, # SW
    (0, -1):  6, # W
    (-1, -1): 7, # NW
}
inverse_directions = {v: k for k, v in directions.items()}

knight_map = {
    (2, 1): 0,
    (2, -1): 1,
    (-2, 1): 2,
    (-2, -1): 3,
    (1, 2): 4,
    (1, -2): 5,
    (-1, 2): 6,
    (-1, -2): 7
}
inverse_knight_map = {v: k for k, v in knight_map.items()}

promoted_map = {
    Pieces.N: 0,
    Pieces.B: 1,
    Pieces.R: 2,
    Pieces.n: 0,
    Pieces.b: 1,
    Pieces.r: 2,
}
inverse_promoted_map = {
    0: Pieces.N,
    1: Pieces.B,
    2: Pieces.R,
}

# see src/framework/chess/board/board.h to see board representation in C
def board_to_tensor(board: Board):
    tensor = torch.zeros(18, 8, 8)
    for i in range(12):
        for r in range(8):
            for f in range(8):
                if get_bit(board.bitboards[i], r * 8 + f):
                    tensor[i][r][f] = 1
    if board.side == 0: tensor[12, :, :] = 1
    else: tensor[12, :, :] = 0
    # TODO
    if board.enpassant != 64:
        tensor[13, board.enpassant // 8, board.enpassant % 8] = 1
    if board.castle & 1: tensor[14, :, :] = 1
    if board.castle & 2: tensor[15, :, :] = 1
    if board.castle & 4: tensor[16, :, :] = 1
    if board.castle & 8: tensor[17, :, :] = 1
    return tensor.unsqueeze(0)


def policy_index_to_move(policy_idx):
    source = policy_idx // 73
    sr, sf = divmod(source, 8)
    move_type = policy_idx % 73
    promoted_piece = -1

    # queen move
    if move_type < 56:
        direction = move_type // 8
        num_sq = move_type % 7 + 1
        dr, df = inverse_directions[direction]
        tr, tf = sr + dr * num_sq, sf + df * num_sq
        target = tr * 8 + tf
    # knight move
    elif move_type < 64:
        direction = move_type - 56
        dr, df = inverse_knight_map[direction]
        tr, tf = sr + dr, sf + df
        target = tr * 8 + tf
    # underpromotion
    else:
        df = ((move_type - 64) % 3) - 1
        promoted_piece = inverse_promoted_map[(move_type - 64) // 3]
        # white pawn promoted
        if source - 16 <= 0:
            tr = 0
        else:
            tr = 7
            # +6 makes Pieces.N -> Pieces.n
            promoted_piece += 6
        target = tr * 8 + sf + df

    return source, target, promoted_piece 


# https://open.spotify.com/track/5cZqsjVs6MevCnAkasbEOX?si=51e94e8fac42445c
def move_to_policy_index(move):
    source, target, promoted = get_move_source(move), get_move_target(move), get_move_promoted(move)
    sr, sf = divmod(source, 8)
    tr, tf = divmod(target, 8)
    dr, df = tr - sr, tf - sf

    if promoted and promoted != Pieces.Q and promoted != Pieces.q:
        # left, center, right  (from white's perspective)
        move_dir = 1 if df == 0 else 2 if df > 0 else 0
        return source * 73 + 64 + promoted_map[promoted] * 3 + move_dir

    if (abs(dr) == 1 and abs(df) == 2) or (abs(dr) == 2 and abs(df) == 1):
        return source * 73 + 56 + knight_map[(dr, df)]

    dr_norm = 1 if dr > 0 else -1 if dr < 0 else 0
    df_norm = 1 if df > 0 else -1 if df < 0 else 0
    if dr_norm == df_norm == 0:
        logging.warning("uhhh yeah let me move my pawn from a2 to a2")
        return -1
    dir_idx = directions[(dr_norm, df_norm)]
    num_sq = max(abs(dr), abs(df))

    return 73 * source + (dir_idx * 7) + num_sq - 1


class Node:
    def __init__(self, player, prior=1.0, parent=None, move=None):
        self.player, self.parent, self.move = player, parent, move
        self.children = {}
        self.visit_count, self.total_value, self.prior, self.Q = 0, 0.0, prior, 0.0
        self.is_expanded = False

    def is_terminal(self, board):
        moves = Moves()
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        return moves.count == 0

    def expand(self, board, policy_tensor):
        moves = Moves()
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        for i in range(moves.count):
            move = moves.moves[i]
            policy_idx = move_to_policy_index(move)
            prior = policy_tensor[policy_idx].item()
            self.children[move] = Node(1 - self.player, prior, self, move)
        self.is_expanded = True


# TODO: passing my mcts tests relies very much on c_puct being set to i have no idea. 
#       Gotta implement a way to tune this while doing a search or something
def select_child(node, c_puct=4.0):
    best_child = None
    best_score = -float('inf')
    # if node.move and move_to_str(node.move) == "e7h4":
    #     print(f"--- select_child for move {move_to_str(node.move) if node.move else "root":6}---")
    for child in node.children.values():
        U = c_puct * child.prior * (node.visit_count**0.5 / (1 + child.visit_count))
        score = child.Q + U
        # if node.move and move_to_str(node.move) == "e7h4":
        #     print(f"    move={move_to_str(child.move):6}  Q={child.Q: .3f}  U={U: .3f}   score={score: .3f}")
        if score > best_score:
            best_score = score
            best_child = child
    # print(f" selected ->{move_to_str(best_child.move):6} (score {best_score:.3f})")
    return best_child


def mcts_search(root_state, net, config: Config, device, add_noise=False):
    root = Node(root_state.side)

    policy, value = net(board_to_tensor(root_state).to(device))
    policy = torch.softmax(policy, dim=1)
    root.expand(root_state, policy[0])

    for i in range(config.mcts.num_simulations):
        node, path = root, [root]
        board = Board()
        copy_board(ctypes.byref(root_state), ctypes.byref(board))

        while node.is_expanded and not node.is_terminal(board):
            node = select_child(node)
            if node is None:
                logging.warning("here 2")
                break
            make_move(ctypes.byref(board), node.move)
            path.append(node)

        if node and node.is_terminal(board):
            value = outcome_value(board)
        elif node:
            policy, value = net(board_to_tensor(board).to(device))
            policy = torch.softmax(policy, dim=1)
            node.expand(board, policy[0])

        if node:
            v = value.item()
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += v
                v = -v
                n.Q = n.total_value / n.visit_count

            # if i % 1000 == 999:
            #     print(f"\n\n{bcolors.FAIL}------- Iteration {i+1} -------{bcolors.ENDC}")
            #     for move, child in root.children.items():
            #         print(f"    move={move_to_str(move):6}  visits={child.visit_count:4d}   Q={child.Q: .3f}")
    
    return root


def calculate_material_advantage(board: Board):
    piece_values = {
        0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 0,  # PNBRQK
        6: 1, 7: 3, 8: 3, 9: 5, 10: 9, 11: 0 # pnbrqk
    }
    white_material = 0
    black_material = 0

    for piece_type in range(12):
        bb = board.bitboards[piece_type]
        count = 0
        for i in range(64):
            if get_bit(bb, i):
                count += 1
        
        if piece_type < 6: # white
            white_material += count * piece_values[piece_type]
        else: # black
            black_material += count * piece_values[piece_type]

    if board.side == 0: # white's move
        return (white_material - black_material) / (white_material + black_material + 1e-6) # normalize
    else: # black
        return (black_material - white_material) / (white_material + black_material + 1e-6)


def outcome_value(state):
    moves = Moves()
    generate_moves(ctypes.byref(state), ctypes.byref(moves))
    if moves.count == 0:
        if is_in_check(ctypes.byref(state), state.side):
            return torch.tensor([1.0], dtype=torch.float32) # checkmate TODO
        else:
            return torch.tensor([0.0], dtype=torch.float32) # stalemate
    return torch.tensor([0.0], dtype=torch.float32) # throw error?


if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from resnet import ResNet6
    init_all_attack_tables()
    board = Board()
    parse_fen(ctypes.byref(board), b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    net = ResNet6().to(device)
    net = torch.compile(net)  # TODO: what the hell is the problem with torch.compile and why does everything break
    start = time.time()
    root = mcts_search(board, net, config, device)
    end = time.time()
    sims_per_sec = config.mcts.num_simulations / (end - start) if end > start else 0
    logging.info(f"MCTS: {config.mcts.num_simulations} sims in {end - start:.2f}s -> {sims_per_sec:.0f} sims/sec")

