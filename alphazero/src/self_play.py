import torch
import ctypes
import time
import os
import random
from datetime import datetime
import logging
import argparse
import multiprocessing
from functools import partial
from resnet import ResNet6
import mcts
from wrapper import (
    Board, Moves,
    init_all_attack_tables, parse_fen, make_move, generate_moves, is_in_check,
    get_move_source, get_move_target, get_move_promoted
)
from config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def square_to_str(sq):
    f = "abcdefgh"[sq % 8]
    r = "12345678"[sq // 8]
    return f + r

def move_to_str(move):
    source_sq = get_move_source(move)
    target_sq = get_move_target(move)
    promoted_piece = get_move_promoted(move)

    move_str = square_to_str(source_sq) + square_to_str(target_sq)

    if promoted_piece:
        promoted_char = {1: 'n', 2: 'b', 3: 'r', 4: 'q',
                         7: 'n', 8: 'b', 9: 'r', 10: 'q'}.get(promoted_piece, '')
        move_str += promoted_char
    return move_str

def set_seeds(seed=69):
    random.seed(seed)
    torch.manual_seed(seed)

def play_game(config: Config, device_str: str, model_path: str, _):
    device = torch.device(device_str)
    net = ResNet6().to(device)
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # compiled torch model has _orig_mod. prepended to keys
        if next(iter(state_dict)).startswith('_orig_mod.'):
            state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
    net.eval()

    init_all_attack_tables()
    board = Board()
    parse_fen(ctypes.byref(board), b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    game_history = []
    moves = Moves()
    move_count = 0
    outcome = 0.0

    was_decided = False
    for depth in range(config.self_play.depth_limit):
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        if moves.count == 0:
            if is_in_check(ctypes.byref(board), board.side):
                outcome = -1.0
            else:
                outcome = 0.0
            was_decided = True
            break

        root = mcts.mcts_search(board, net, config, device, add_noise=True)

        policy = torch.zeros(4672)
        for move, child in root.children.items():
            policy_idx = mcts.move_to_policy_index(move, board)
            policy[policy_idx] = child.visit_count
        if torch.sum(policy) > 0:
            policy /= torch.sum(policy)

        game_history.append([
            mcts.board_to_tensor(board).detach().cpu(),
            policy.detach().cpu(),
            None
        ])

        if move_count < config.self_play.temp_threshold:
            move_probs = torch.tensor([c.visit_count for c in root.children.values()], dtype=torch.float32, device='cpu')
            if torch.sum(move_probs) > 0:
                move_probs /= torch.sum(move_probs)
                selected_move_idx = torch.multinomial(move_probs, 1).item()
                selected_move = list(root.children.keys())[selected_move_idx]
            else:
                selected_move = random.choice(list(root.children.keys()))
        selected_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)

        policy_idx = mcts.move_to_policy_index(selected_move, board)
        legal_moves = [move_to_str(m) for m in root.children.keys()]
        # logging.info(f"Turn {move_count + 1}: net picked index {policy_idx} -> move: {move_to_str(selected_move)}")
        # logging.info(f"         legal: {legal_moves}")
        # logging.info(f"         selected: {move_to_str(selected_move)}")

        make_move(ctypes.byref(board), selected_move)
        move_count += 1

    if not was_decided:
        outcome = mcts.calculate_material_advantage(board)

    for i in range(len(game_history)):
        game_history[i][2] = outcome 

    return game_history

def run_self_play(config: Config, output_dir: str, device: torch.device):
    set_seeds()
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    if not os.path.exists(best_model_path):
        logging.info("randomly creating best model")
        initial_net = ResNet6().to(device)
        torch.save(initial_net.state_dict(), best_model_path)

    start_time = time.time()
    data_dir = os.path.join(output_dir, 'self_play_data')
    os.makedirs(data_dir, exist_ok=True)

    game_func = partial(play_game, config, str(device), best_model_path)

    with multiprocessing.Pool(processes=24) as pool:
        desc = "Playing training games"
        game_iterator = pool.imap_unordered(game_func, range(config.self_play.num_games))
        all_training_data = list(tqdm(game_iterator, total=config.self_play.num_games, desc=f"{desc:>30}", colour="green"))

    for i, training_data in enumerate(all_training_data):
        file_path = os.path.join(data_dir, f"game_{start_time}_{i:06d}.pt")
        torch.save(training_data, file_path)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug end to end that it runs")
    args = parser.parse_args()

    config = Config(debug=args.debug)
    if args.debug:
        logging.info(" DEBUGGING ")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_self_play(config, run_output_dir, device)


