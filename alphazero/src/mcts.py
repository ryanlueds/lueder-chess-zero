import time
import ctypes
import torch
import logging
from wrapper import (
    Board, Moves,
    init_all_attack_tables, parse_fen,
    generate_moves, make_move, copy_board,
    get_move_source, get_move_target, get_move_promoted,
    get_bit, is_in_check
)
from config import Config

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

# only god knows what's happening here. sorry future ryan
def move_to_policy_index(move, board):
    source, target, promoted = get_move_source(move), get_move_target(move), get_move_promoted(move)
    sr, sf = divmod(source, 8)
    tr, tf = divmod(target, 8)
    dr, df = tr - sr, tf - sf

    if promoted and promoted % 6 != 4:
        promoted_map = {1: 0, 2: 1, 3: 2, 7: 0, 8: 1, 9: 2}
        move_dir = 1 if df == 0 else 2 if (board.side == 0 and df > 0) or (board.side == 1 and df < 0) else 0
        return source * 73 + 64 + promoted_map[promoted] * 3 + move_dir

    if (abs(dr) == 1 and abs(df) == 2) or (abs(dr) == 2 and abs(df) == 1):
        knight_map = {(2, 1): 0, (2, -1): 1, (-2, 1): 2, (-2, -1): 3, (1, 2): 4, (1, -2): 5, (-1, 2): 6, (-1, -2): 7}
        return source * 73 + 56 + knight_map[(dr, df)]

    if dr == 0 and df == 0:
        logging.warning("Here 1")
        return -1  # https://open.spotify.com/track/5cZqsjVs6MevCnAkasbEOX?si=51e94e8fac42445c

    s_dr = 1 if dr > 0 else -1 if dr < 0 else 0
    s_df = 1 if df > 0 else -1 if df < 0 else 0
    direction_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
    return source * 73 + direction_map[(s_dr, s_df)] * 7 + max(abs(dr), abs(df)) - 1


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
            policy_idx = move_to_policy_index(move, board)
            prior = torch.exp(policy_tensor[policy_idx]).item()
            self.children[move] = Node(1 - self.player, prior, self, move)
        self.is_expanded = True


def select_child(node, c_puct=2.0):
    best_child = None
    best_score = -float('inf')
    for child in node.children.values():
        score = child.Q + c_puct * child.prior * (node.visit_count**0.5 / (1 + child.visit_count))
        if score > best_score:
            best_score = score
            best_child = child
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
            node.expand(board, policy[0])

        if node:
            v = value.item()
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += v
                v = -v
                n.Q = n.total_value / n.visit_count
    
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
            return torch.tensor([-1.0], dtype=torch.float32) # checkmate TODO
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

