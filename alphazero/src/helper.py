import torch
import torch.nn as nn
import unittest
import ctypes
import sys
import os
from enum import IntEnum
import inspect
from resnet import ResNet6
from config import Config
import mcts
from wrapper import (
    Board, Moves,
    init_all_attack_tables, parse_fen, make_move, generate_moves, is_in_check,
    get_move_source, get_move_target, get_move_promoted
)

def square_to_str(sq):
    rank = sq // 8
    file = sq % 8
    real_rank = 7 - rank
    real_sq = real_rank * 8 + file
    f = "abcdefgh"[real_sq % 8]
    r = "12345678"[real_sq // 8]
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


def find_best_move(fen, num_simulations, net):
    config = Config()
    config.mcts.num_simulations = num_simulations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = Board()
    parse_fen(ctypes.byref(board), fen.encode('utf-8'))

    root = mcts.mcts_search(board, net, config, device)
    if not root.children:
        return None

    sorted_children = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
    
    best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
    return move_to_str(best_move)


def convert_channel_to_bitboard(tensor_channel):
    bitboard = 0
    for r in range(8):
        for f in range(8):
            if tensor_channel[r][f] == 1:
                bitboard |= (1 << (r * 8 + f))
    return bitboard
