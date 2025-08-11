import chess
import chess.pgn
import chess.engine
import os
import glob
import json
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_stockfish_analysis(board, engine, num_moves=5):
    """Get the top N moves and their evaluations from Stockfish."""
    info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=num_moves)
    top_moves = []
    for i in range(len(info)):
        move = info[i]["pv"][0]
        score = info[i]["score"].white().score(mate_score=10000)
        top_moves.append({"move": move.uci(), "score": score})
    
    # Get the overall position evaluation from the first principal variation
    position_evaluation = info[0]["score"].white().score(mate_score=10000)
    
    return top_moves, position_evaluation

def annotate_games(config, args):
    stockfish_path = os.path.join(os.path.dirname(__file__), '..', 'bin', 'stockfish-ubuntu-x86-64-avx2')
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    logging.info("wakey wakey stockfish")

    pgn_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'filtered_chess_games')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'chess_games', 'annotated_chess_games')
    os.makedirs(output_dir, exist_ok=True)

    pgn_files = glob.glob(os.path.join(pgn_dir, '*.pgn'))
    if args.debug:
        pgn_files = pgn_files[:1]
        logging.warning("DEBUG: Using a single PGN file.")

    for pgn_file in tqdm(pgn_files, desc="Annotating PGNs", colour="green"):
        annotated_data = []
        output_filename = os.path.basename(pgn_file).replace('.pgn', '.json')
        output_path = os.path.join(output_dir, output_filename)

        with open(pgn_file) as pgn:
            games = iter(lambda: chess.pgn.read_game(pgn), None)
            with tqdm(games, desc=f"Processing {os.path.basename(pgn_file)}", unit="games", leave=False, colour="blue") as pbar:
                for game in pbar:
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

        with open(output_path, 'w') as json_file:
            json.dump(annotated_data, json_file, indent=2)
        
        logging.info(f"Finished annotating {os.path.basename(pgn_file)} and saved to {output_filename}")

    engine.quit()
    logging.info("Annotation process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate PGN files with Stockfish evaluations.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to process only one PGN file.")
    args = parser.parse_args()
    
    from config import Config
    config = Config()
    
    annotate_games(config, args)
