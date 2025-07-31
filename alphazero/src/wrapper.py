import ctypes
from ctypes import (
    c_uint64, c_int, Structure, CDLL, POINTER,
    c_char_p, byref
)
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "../../build/lib/libchess_engine.so")
chess_lib = CDLL(LIB_PATH)

class Board(Structure):
    _fields_ = [
        ("bitboards", c_uint64 * 12),
        ("occupancies", c_uint64 * 3),
        ("side", c_int),
        ("enpassant", c_int),
        ("castle", c_int),
    ]

class Moves(Structure):
    _fields_ = [
        ("moves", c_int * 256),
        ("count", c_int),
    ]

class MoveData(Structure):
    _fields_ = [
        ("move", c_int),
        ("captured_piece", c_int),
        ("castling_rights", c_int),
        ("enpassant_square", c_int),
        ("occupancies", c_uint64 * 3),
    ]


chess_lib.init_all_attack_tables.argtypes = []
chess_lib.init_all_attack_tables.restype = None

chess_lib.parse_fen.argtypes = [POINTER(Board), c_char_p]
chess_lib.parse_fen.restype = None

chess_lib.generate_moves.argtypes = [POINTER(Board), POINTER(Moves)]
chess_lib.generate_moves.restype = None

chess_lib.make_move.argtypes = [POINTER(Board), c_int]
chess_lib.make_move.restype = c_int

chess_lib.undo_move.argtypes = [POINTER(Board), POINTER(MoveData)]
chess_lib.undo_move.restype = None

chess_lib.get_move_source_wrapper.argtypes = [c_int]
chess_lib.get_move_source_wrapper.restype = c_int

chess_lib.get_move_target_wrapper.argtypes = [c_int]
chess_lib.get_move_target_wrapper.restype = c_int

chess_lib.get_move_promoted_wrapper.argtypes = [c_int]
chess_lib.get_move_promoted_wrapper.restype = c_int

chess_lib.get_move_piece_wrapper.argtypes = [c_int]
chess_lib.get_move_piece_wrapper.restype = c_int

chess_lib.get_move_capture_wrapper.argtypes = [c_int]
chess_lib.get_move_capture_wrapper.restype = c_int

chess_lib.get_move_enpassant_wrapper.argtypes = [c_int]
chess_lib.get_move_enpassant_wrapper.restype = c_int

chess_lib.get_bit_wrapper.argtypes = [c_uint64, c_int]
chess_lib.get_bit_wrapper.restype = c_uint64

chess_lib.copy_board.argtypes = [POINTER(Board), POINTER(Board)]
chess_lib.copy_board.restype = None

chess_lib.take_back.argtypes = [POINTER(Board), POINTER(Board)]
chess_lib.take_back.restype = None

chess_lib.is_in_check_wrapper.argtypes = [POINTER(Board), c_int]
chess_lib.is_in_check_wrapper.restype = c_uint64

__all__ = [
    "Board", "Moves", "MoveData", "init_all_attack_tables", "parse_fen",
    "generate_moves", "get_move_source", "get_move_target", "get_move_promoted",
    "get_move_piece", "get_move_capture", "get_move_enpassant",
    "get_bit", "make_move", "undo_move", "copy_board", "take_back", "is_in_check", "get_move_double"
]

init_all_attack_tables = chess_lib.init_all_attack_tables
parse_fen = chess_lib.parse_fen
generate_moves = chess_lib.generate_moves
make_move = chess_lib.make_move
undo_move = chess_lib.undo_move
copy_board = chess_lib.copy_board
take_back = chess_lib.take_back
is_in_check = chess_lib.is_in_check_wrapper

get_move_source = chess_lib.get_move_source_wrapper
get_move_target = chess_lib.get_move_target_wrapper
get_move_promoted = chess_lib.get_move_promoted_wrapper
get_move_piece = chess_lib.get_move_piece_wrapper
get_move_capture = chess_lib.get_move_capture_wrapper
get_move_enpassant = chess_lib.get_move_enpassant_wrapper
get_bit = chess_lib.get_bit_wrapper
get_move_double = chess_lib.get_move_double_wrapper
