import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import chess.pgn
import json
from resnet import ResNet6
from config import Config
from utils import move_to_str
from wrapper import Board, parse_fen, generate_moves, Moves
from mcts import board_to_tensor, move_to_policy_index
import os
import glob
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import ctypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, device, optimizer, data_loader):
    model.train()
    total_policy_loss = 0
    total_value_loss = 0

    for states, policies, values in tqdm(data_loader, desc="Training Epoch", colour="cyan", leave=False):
        states, policies, values = states.to(device), policies.to(device), values.to(device)

        optimizer.zero_grad()
        policy_logits, value_preds = model(states)

        policy_loss = torch.nn.functional.cross_entropy(policy_logits, policies)
        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(-1), values)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    num_batches = len(data_loader)
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    return avg_policy_loss, avg_value_loss

def plot_losses(output_dir, policy_losses, value_losses):
    plt.figure()
    epochs = range(1, len(policy_losses) + 1)
    plt.plot(epochs, policy_losses, 'b-o', label='Policy Loss')
    plt.plot(epochs, value_losses, 'r-o', label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'bootstrap_loss_graph.png'))
    plt.close()

def scores_to_probabilities(top_moves, device):
    scores = torch.tensor([move['score'] for move in top_moves], dtype=torch.float32, device=device)
    # Normalize scores to be in a reasonable range for softmax
    # Dividing by 100 converts centipawns to pawn units.
    scaled_scores = scores / 100.0
    return torch.nn.functional.softmax(scaled_scores, dim=0)

def run_bootstrap_training(config: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', f"bootstrap_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Run directory created at: {run_dir}")

    json_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'annotated_chess_games')
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    if args.debug:
        json_files = json_files[:1]
        logging.warning("DEBUG mode enabled: Using a single JSON file.")

    all_states, all_policies, all_values = [], [], []
    max_positions = 500 if args.debug else 50000

    for json_file in tqdm(json_files, desc="Processing JSON files", colour="green"):
        if len(all_states) >= max_positions:
            break
        with open(json_file) as f:
            annotated_data = json.load(f)
            for data in annotated_data:
                if len(all_states) >= max_positions:
                    break
                
                c_board = Board()
                parse_fen(ctypes.byref(c_board), data['fen'].encode('utf-8'))
                state_tensor = board_to_tensor(c_board)

                top_moves = data.get('top_moves')
                if not top_moves:
                    continue

                probabilities = scores_to_probabilities(top_moves, device)
                policy_target = torch.zeros(4672, device=device)

                legal_moves = Moves()
                generate_moves(ctypes.byref(c_board), ctypes.byref(legal_moves))
                legal_move_map = {move_to_str(legal_moves.moves[i]): legal_moves.moves[i] for i in range(legal_moves.count)}

                for i, move_data in enumerate(top_moves):
                    move_str = move_data['move']
                    if move_str in legal_move_map:
                        move = legal_move_map[move_str]
                        policy_idx = move_to_policy_index(move)
                        if policy_idx != -1:
                            policy_target[policy_idx] = probabilities[i]
                
                if torch.sum(policy_target) > 0:
                    policy_target /= torch.sum(policy_target)
                    all_states.append(state_tensor)
                    all_policies.append(policy_target.unsqueeze(0))
                    all_values.append(torch.tensor([max(-1.0, min(1.0, data['position_evaluation'] / 1000.0))], dtype=torch.float32, device=device))

    if not all_states:
        logging.error("No positions were processed. Exiting.")
        return

    dataset = TensorDataset(
        torch.cat(all_states),
        torch.cat(all_policies),
        torch.cat(all_values)
    )
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = ResNet6().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    policy_losses, value_losses = [], []

    # TODO: Train for multiple epochs
    avg_policy_loss, avg_value_loss = train_epoch(model, device, optimizer, data_loader)
    policy_losses.append(avg_policy_loss)
    value_losses.append(avg_value_loss)
    logging.info(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

    plot_losses(run_dir, policy_losses, value_losses)
    logging.info(f"Saved loss graph to {run_dir}")
    
    model_path = os.path.join(run_dir, 'bootstrap_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config()
    run_bootstrap_training(config, args)
