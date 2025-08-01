import zstandard
import chess.pgn
import argparse
import os
import logging
from io import TextIOWrapper
from glob import glob
import multiprocessing


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


def is_game_valid(game, min_elo):
    white_elo_str = game.headers.get("WhiteElo", "0")
    black_elo_str = game.headers.get("BlackElo", "0")

    try:
        white_elo = int(white_elo_str)
        black_elo = int(black_elo_str)
    except ValueError:
        return False

    if white_elo < min_elo or black_elo < min_elo:
        return False

    time_control = game.headers.get("TimeControl", "")
    if "60+0" in time_control or "60+1" in time_control:    # bullet
        return False
    if "180+0" in time_control or "180+2" in time_control:  # blitz
        return False

    event = game.headers.get("Event", "").lower()
    if "blitz" in event or "bullet" in event:
        return False

    return True


def process_pgn_file(input_path, output_path, min_elo=2000):
    with open(input_path, "r", encoding='utf-8') as infile, open(output_path, "w", encoding='utf-8') as outfile:
        game_count = 0
        filtered_count = 0
        while True:
            try:
                game = chess.pgn.read_game(infile)
                if game is None:
                    break

                game_count += 1
                if is_game_valid(game, min_elo):
                    outfile.write(str(game) + "\n\n")
                    filtered_count += 1

            except Exception as e:
                logging.warning(f"Error reading game: {e}")
                continue

        logging.info(f"{input_path}: Processed {game_count} games, kept {filtered_count}")


def parallel_process_all(input_dir, output_dir, min_elo=2000):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, "lichess_part_*"))

    args = []
    for path in input_files:
        out_path = os.path.join(output_dir, os.path.basename(path) + ".filtered.pgn")
        args.append((path, out_path, min_elo))

    with multiprocessing.Pool(processes=24) as pool:
        pool.starmap(process_pgn_file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    min_elo = 2000
    decompress_and_filter_pgn(args.input_file, args.output_file, min_elo)

