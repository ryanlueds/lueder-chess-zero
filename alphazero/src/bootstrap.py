# bootstrap_train.py
import os
import glob
import json
import ctypes
import logging
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from resnet import ResNet6
from config import Config
from utils import move_to_str
from wrapper import Board, parse_fen, generate_moves, Moves
from mcts import board_to_tensor, move_to_policy_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_epoch(model, device, optimizer, data_loader):
    """
    Single training epoch. Uses KL-divergence for policy (probability distributions)
    and MSE for value. Expects:
      - policy targets: (N, C) float probabilities (sum to 1 per row)
      - value targets: (N,) or (N,1)
    """
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for states, policies, values in tqdm(data_loader, desc="Training Epoch", colour="cyan", leave=False):
        states = states.to(device)
        policies = policies.to(device, dtype=torch.float32)
        values = values.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        policy_logits, value_preds = model(states)

        # Policy loss: KLDiv between target distribution (policies) and model distribution.
        # torch.nn.functional.kl_div expects input = log-probs, target = probs.
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = torch.nn.functional.kl_div(log_probs, policies, reduction='batchmean')

        # Value loss: safe reshape to 1D
        value_loss = torch.nn.functional.mse_loss(value_preds.view(-1), values.view(-1))

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        total_policy_loss += float(policy_loss.detach().cpu().item())
        total_value_loss += float(value_loss.detach().cpu().item())
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    return avg_policy_loss, avg_value_loss


def plot_losses(output_dir, policy_losses, value_losses):
    plt.figure()
    epochs = list(range(1, len(policy_losses) + 1))
    plt.plot(epochs, policy_losses, 'b-o', label='Policy Loss')
    plt.plot(epochs, value_losses, 'r-o', label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    outpath = os.path.join(output_dir, 'bootstrap_loss_graph.png')
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Saved loss graph to {outpath}")


def scores_to_probabilities(top_moves, is_black_turn, device):
    """
    Convert move scores (centipawns) to a probability distribution using softmax.
    Caller guaranteed that scores are centipawn numeric values (no mate tokens).
    If it's black to move, negate scores so higher is better for moving side.
    """
    scores = torch.tensor([move['score'] for move in top_moves], dtype=torch.float32, device=device)
    if is_black_turn:
        scores = -scores
    # Scale down to stabilize softmax (centipawns -> pawn fractions)
    scaled = scores / 100.0
    probs = torch.nn.functional.softmax(scaled, dim=0)
    return probs


def run_bootstrap_training(config: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', f"bootstrap_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Run directory created at: {run_dir}")

    # JSON annotated games directory
    json_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'annotated_chess_games')
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    if args.debug:
        json_files = json_files[:1]
        logging.warning("DEBUG mode enabled: Using a single JSON file.")

    model = ResNet6().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    policy_losses = []
    value_losses = []

    cpu_device = torch.device("cpu")
    num_workers = 24
    logging.info(f"Using {num_workers} workers for data loading.")

    num_epochs = 1 if not args.debug else 1
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}/{num_epochs}")
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        files_processed_in_epoch = 0

        file_iterator = tqdm(sorted(json_files), desc=f"Epoch {epoch}/{num_epochs}", colour="green")
        for json_file in file_iterator:
            all_states = []
            all_policies = []
            all_values = []

            try:
                with open(json_file, 'r') as f:
                    annotated_data = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load {json_file}: {e}")
                continue

            for data in annotated_data:
                # Skip very early positions: optional but kept from original logic
                try:
                    # Fullmove number is the 6th token in FEN; fallback if not present
                    fen_tokens = data.get('fen', '').split(' ')
                    if len(fen_tokens) >= 6:
                        fullmove_number = int(fen_tokens[5])
                    else:
                        # If format differs, try last token
                        fullmove_number = int(fen_tokens[-1])
                except Exception:
                    fullmove_number = 1

                if fullmove_number < 5:
                    continue

                # Prepare board via C wrapper
                c_board = Board()
                try:
                    parse_fen(ctypes.byref(c_board), data['fen'].encode('utf-8'))
                except Exception as e:
                    logging.debug(f"parse_fen failed for fen {data.get('fen')}: {e}")
                    continue

                # Convert board to tensor (assumed CPU tensor)
                try:
                    state_tensor = board_to_tensor(c_board)  # expecting a torch tensor
                except Exception as e:
                    logging.debug(f"board_to_tensor failed: {e}")
                    continue

                top_moves = data.get('top_moves')
                if not top_moves:
                    continue

                is_black_turn = (c_board.side == 1)
                probabilities = scores_to_probabilities(top_moves, is_black_turn, cpu_device)

                # Build a zeroed policy vector per position
                policy_target = torch.zeros(4672, dtype=torch.float32, device=cpu_device)

                # Generate legal moves from C wrapper and map string->move
                legal_moves = Moves()
                try:
                    generate_moves(ctypes.byref(c_board), ctypes.byref(legal_moves))
                except Exception as e:
                    logging.debug(f"generate_moves failed: {e}")
                    continue

                legal_move_map = {}
                for i in range(getattr(legal_moves, 'count', 0)):
                    try:
                        mstr = move_to_str(legal_moves.moves[i])
                        legal_move_map[mstr] = legal_moves.moves[i]
                    except Exception:
                        continue

                # Map Stockfish top_moves to indices and fill policy_target
                for i, move_data in enumerate(top_moves):
                    move_str = move_data.get('move')
                    if move_str is None:
                        continue
                    if move_str in legal_move_map:
                        move = legal_move_map[move_str]
                        policy_idx = move_to_policy_index(move)
                        if policy_idx != -1:
                            # probabilities is on cpu_device and dtype float32
                            try:
                                policy_target[policy_idx] = float(probabilities[i].detach().cpu().item())
                            except Exception:
                                # fallback if probabilities[i] isn't tensor-like
                                policy_target[policy_idx] = float(probabilities[i])

                if torch.sum(policy_target) <= 0.0:
                    # No overlap between annotated moves and legal moves => skip
                    continue

                # Normalize to make a probability distribution
                policy_target = policy_target / torch.sum(policy_target)

                # Value handling: position evaluation is from white POV in dataset
                stockfish_eval_white_pov = data.get('position_evaluation', None)
                if stockfish_eval_white_pov is None:
                    continue
                value = stockfish_eval_white_pov
                if c_board.side == 1:
                    value = -value
                normalized_value = max(-1.0, min(1.0, value / 1000.0))

                # Append tensors (keep on CPU for DataLoader to handle transfer)
                # Ensure state_tensor is a torch.Tensor on CPU
                if isinstance(state_tensor, torch.Tensor):
                    st_cpu = state_tensor.detach().cpu()
                else:
                    # If board_to_tensor returned numpy, convert
                    st_cpu = torch.tensor(state_tensor, dtype=torch.float32)

                all_states.append(st_cpu)  # board_to_tensor already returns a batched tensor
                all_policies.append(policy_target.unsqueeze(0))  # shape (1, 4672)
                all_values.append(torch.tensor([[normalized_value]], dtype=torch.float32))  # shape (1,1)

            if not all_states:
                logging.warning(f"No positions were processed from {json_file}. Skipping.")
                continue

            # Concatenate everything
            states_tensor = torch.cat(all_states, dim=0)
            policies_tensor = torch.cat(all_policies, dim=0)
            values_tensor = torch.cat(all_values, dim=0).view(-1)

            dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)

            data_loader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if device.type == 'cuda' else False
            )

            avg_policy_loss, avg_value_loss = train_epoch(model, device, optimizer, data_loader)
            if avg_policy_loss > 0 or avg_value_loss > 0:
                epoch_policy_loss += avg_policy_loss
                epoch_value_loss += avg_value_loss
                files_processed_in_epoch += 1

        if files_processed_in_epoch > 0:
            avg_epoch_policy_loss = epoch_policy_loss / files_processed_in_epoch
            avg_epoch_value_loss = epoch_value_loss / files_processed_in_epoch
            policy_losses.append(avg_epoch_policy_loss)
            value_losses.append(avg_epoch_value_loss)
            logging.info(f"Epoch {epoch} - Policy Loss: {avg_epoch_policy_loss:.6f}, Value Loss: {avg_epoch_value_loss:.6f}")
        else:
            logging.warning(f"No data processed in epoch {epoch}.")

    # Plot and save
    if not policy_losses and not value_losses:
        logging.error("No training was performed, nothing to plot or save.")
        return

    plot_losses(run_dir, policy_losses, value_losses)

    model_path = os.path.join(run_dir, 'bootstrap_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config()
    run_bootstrap_training(config, args)
