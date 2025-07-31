from wrapper import (
    Board, Moves, MoveData,
    init_all_attack_tables, parse_fen,
    generate_moves, make_move, undo_move,
    copy_board, take_back,
    get_move_capture, get_move_target, get_move_enpassant, get_bit
)
import time
import ctypes
from enum import IntEnum


class Pieces(IntEnum):
    P = 0
    N = 1
    B = 2
    R = 3
    Q = 4
    K = 5
    p = 6
    n = 7
    b = 8
    r = 9
    q = 10
    k = 11

class Sides(IntEnum):
    white = 0
    black = 1
    both = 2

class Mode(IntEnum):
    all_moves = 0
    only_captures = 1


start_position = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def perft_driver(board: Board, depth: int) -> int:
    move_list = Moves()
    generate_moves(ctypes.byref(board), ctypes.byref(move_list))
    data = MoveData()

    nodes = 0

    if depth == 1:
        return move_list.count
    
    # this is just ported from src/test/perft.h
    # but this one is 30x slower i dont get it
    for i in range(0, move_list.count):
        data.move = move_list.moves[i]
        data.castling_rights = board.castle
        data.enpassant_square = board.enpassant

        data.occupancies[0] = board.occupancies[0]
        data.occupancies[1] = board.occupancies[1]
        data.occupancies[2] = board.occupancies[2]

        data.captured_piece = make_move(ctypes.byref(board), data.move)

        if depth == 2:
            move_list_d2 = Moves()
            generate_moves(ctypes.byref(board), ctypes.byref(move_list_d2))
            nodes += move_list_d2.count
        else:
            nodes += perft_driver(board, depth - 1)
        
        undo_move(ctypes.byref(board), ctypes.byref(data))

    return nodes


if __name__ == "__main__":
    start = time.time()
    board = Board()
    init_all_attack_tables()
    parse_fen(ctypes.byref(board), start_position)

    nodes = perft_driver(board, 4)
    end = time.time()

    print(f"Nodes: {nodes}")
    print(f" Time: {end - start:3f}s")

