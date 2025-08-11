import chess
import chess.pgn
import chess.engine
import os
import glob
import json
import logging
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_stockfish_analysis(board, engine, num_moves=5):
    """Get the top N moves and their evaluations from Stockfish."""
    info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=num_moves)
    top_moves = []
    for i in range(len(info)):
        move = info[i]["pv"][0]
        score = info[i]["score"].white().score(mate_score=10000)
        top_moves.append({"move": move.uci(), "score": score})
    
    position_evaluation = info[0]["score"].white().score(mate_score=10000)
    
    return top_moves, position_evaluation

def process_pgn_file(pgn_file, output_dir, stockfish_path, debug=False):
    """Processes a single PGN file, annotates it, and saves the JSON output."""
    try:
        output_filename = os.path.basename(pgn_file).replace('.pgn', '.json')
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            return f"Skipped: {os.path.basename(pgn_file)}"

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        annotated_data = []
        with open(pgn_file) as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        fen = board.fen()
                        top_moves, position_evaluation = get_stockfish_analysis(board, engine, num_moves=5)

                        annotated_data.append({
                            "fen": fen,
                            "played_move": move.uci(),
                            "position_evaluation": position_evaluation,
                            "top_moves": top_moves
                        })

                        board.push(move)
                    
                    if debug:
                        break
                except Exception as e:
                    logging.warning(f"Skipping a malformed game in {os.path.basename(pgn_file)}: {e}")


        with open(output_path, 'w') as json_file:
            json.dump(annotated_data, json_file, indent=2)
        
        engine.quit()
        return f"Annotated: {os.path.basename(pgn_file)}"
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(pgn_file)}: {e}")
        return f"Error: {os.path.basename(pgn_file)}"

def annotate_games(config, args):
    stockfish_path = os.path.join(os.path.dirname(__file__), '..', 'bin', 'stockfish-ubuntu-x86-64-avx2')
    
    pgn_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'filtered_chess_games')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'annotated_chess_games')
    os.makedirs(output_dir, exist_ok=True)

    pgn_files = glob.glob(os.path.join(pgn_dir, '*.pgn'))
    if args.debug:
        pgn_files = pgn_files[:1]
        logging.warning("DEBUG: Using a single PGN file.")

    num_processes = 24
    logging.info(f"Starting annotation with {num_processes} processes...")

    worker_func = partial(process_pgn_file, output_dir=output_dir, stockfish_path=stockfish_path, debug=args.debug)

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(pgn_files), desc="Annotating PGNs", colour="green") as pbar:
            for result in pool.imap_unordered(worker_func, pgn_files):
                pbar.update(1)

    logging.info("Annotation process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate PGN files with Stockfish evaluations.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to process only one PGN file.")
    args = parser.parse_args()
    
    # Config is not used in this version, but kept for consistency
    from config import Config
    config = Config()
    
    annotate_games(config, args)
