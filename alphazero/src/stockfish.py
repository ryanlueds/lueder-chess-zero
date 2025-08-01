import pprint
import chess
import chess.engine
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "../bin/stockfish-ubuntu-x86-64-avx2")
MULTI_PV = 10

def evaluate_position(board, engine, time_limit_sec=1.0):
    info = engine.analyse(board, chess.engine.Limit(time=time_limit_sec), multipv=MULTI_PV)
    best_moves = []
    for move_info in info:
        best_moves.append((move_info["pv"][0].uci(), move_info["score"].relative.score(mate_score=10_000)))
    pprint.pprint(best_moves)


if __name__ == "__main__":
    fen = "r1bqk2r/ppp1bppp/8/5N2/6P1/5P2/PPPPP3/RNBQKB2 b Qkq - 2 6"
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(os.path.expanduser(STOCKFISH_PATH))
    print(board)
    evaluate_position(board, engine, time_limit_sec=1.0)

