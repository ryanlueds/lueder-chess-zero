import torch
import torch.nn as nn
import unittest
import ctypes
import sys
import os
from enum import IntEnum
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from resnet import ResNet6
from config import Config
import mcts
from wrapper import (
    Board, Moves,
    init_all_attack_tables, parse_fen, make_move, generate_moves, is_in_check,
    get_move_source, get_move_target, get_move_promoted
)
from mcts import board_to_tensor, move_to_policy_index, policy_index_to_move

class Pieces(IntEnum):
    P=0
    N=1
    B=2
    R=3
    Q=4
    K=5
    p=6
    n=7
    b=8
    r=9
    q=10
    k=11


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # uniform policy
        policy = torch.full((x.shape[0], 4672), 1/4672, device=x.device)
        value = torch.zeros((x.shape[0], 1), device=x.device)
        return policy, value


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
    """
    bit boards for each piece in starting position
    P 71776119061217280
    N 4755801206503243776
    B 2594073385365405696
    R 9295429630892703744
    Q 576460752303423488
    K 1152921504606846976
    p 65280
    n 66
    b 36
    r 129
    q 8
    k 16
    """
    bitboard = 0
    for r in range(8):
        for f in range(8):
            if tensor_channel[r][f] == 1:
                bitboard |= (1 << (r * 8 + f))
    return bitboard

class TestMCTS(unittest.TestCase):
    def setUp(self):
        init_all_attack_tables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_net = ResNet6().to(self.device)
        self.board = Board()
        parse_fen(self.board, b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


    def test_board_to_tensor(self):
        tensor = board_to_tensor(self.board)
        self.assertEqual(tensor.shape, (1, 18, 8, 8))

        # bitboards for each piece in starting position
        correct_bitboards = {
            Pieces.P: 71776119061217280,
            Pieces.N: 4755801206503243776,
            Pieces.B: 2594073385365405696,
            Pieces.R: 9295429630892703744,
            Pieces.Q: 576460752303423488,
            Pieces.K: 1152921504606846976,
            Pieces.p: 65280,
            Pieces.n: 66,
            Pieces.b: 36,
            Pieces.r: 129,
            Pieces.q: 8,
            Pieces.k: 16
        }

        for piece, correct_bitboard in correct_bitboards.items():
            bitboard = convert_channel_to_bitboard(tensor[0][piece.value])
            self.assertEqual(bitboard, correct_bitboard, f"failed on piece {piece}")

        
    def test_move_to_policy_index(self):
        board = Board()
        parse_fen(ctypes.byref(board), b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        moves = Moves()
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        e2e4_move = -1
        for i in range(moves.count):
            move = moves.moves[i]
            # "best by test" -- bobby
            if get_move_source(move) == 52 and get_move_target(move) == 36:
                e2e4_move = move
                break
        self.assertNotEqual(e2e4_move, -1)
        policy_index = move_to_policy_index(e2e4_move)
        source, target, promoted = policy_index_to_move(policy_index)
        self.assertEqual(promoted, -1)  # not a promotion
        self.assertEqual(source, 52)
        self.assertEqual(target, 36)

        g1f3_move = -1
        for i in range(moves.count):
            move = moves.moves[i]
            # g1f3 knight move
            if get_move_source(move) == 62 and get_move_target(move) == 45:
                g1f3_move = move
                break

        self.assertNotEqual(g1f3_move, -1)
        policy_index = move_to_policy_index(g1f3_move)
        source, target, promoted = policy_index_to_move(policy_index)
        self.assertEqual(promoted, -1)  # not a promotion
        self.assertEqual(source, 62)
        self.assertEqual(target, 45)

        # underpromotion to a knight with advantage
        board = Board()
        parse_fen(ctypes.byref(board), b"8/5P1k/5Q2/6Q1/8/8/8/4K3 w - - 0 1")
        moves = Moves()
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        f7f8N_move = -1
        for i in range(moves.count):
            move = moves.moves[i]
            if get_move_source(move) == 13 and get_move_target(move) == 5 and get_move_promoted(move) == Pieces.N:
                f7f8N_move = move
                break
        self.assertNotEqual(f7f8N_move, -1)
        policy_index = move_to_policy_index(f7f8N_move)
        source, target, promoted = policy_index_to_move(policy_index)
        self.assertEqual(promoted, Pieces.N)  # white knight
        self.assertEqual(source, 13)
        self.assertEqual(target, 5)

        # black pawn captures to the left, underpromoting to a bishop---with advantage
        board = Board()
        parse_fen(ctypes.byref(board), b"k6K/5q2/8/8/8/8/1p6/Q7 b - - 0 1")
        moves = Moves()
        generate_moves(ctypes.byref(board), ctypes.byref(moves))
        b2a1b_move = -1
        for i in range(moves.count):
            move = moves.moves[i]
            if get_move_source(move) == 49 and get_move_target(move) == 56 and get_move_promoted(move) == Pieces.b:
                b2a1b_move = move
                break
        self.assertNotEqual(b2a1b_move, -1)
        policy_index = move_to_policy_index(b2a1b_move)
        source, target, promoted = policy_index_to_move(policy_index)
        self.assertEqual(promoted, Pieces.b)  # black advantage
        self.assertEqual(source, 49)
        self.assertEqual(target, 56)

    def test_dummy_net_self_play(self):
        config = Config()
        config.mcts.num_simulations = 64
        net = DummyNet().to(self.device)

        board = Board()
        parse_fen(ctypes.byref(board), b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        for _ in range(10):
            moves = Moves()
            generate_moves(ctypes.byref(board), ctypes.byref(moves))
            if moves.count == 0:
                break

            root = mcts.mcts_search(board, net, config, self.device)

            # only generates legal moves
            selected_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
            legal_move_keys = [m for m in root.children.keys()]
            self.assertIn(selected_move, legal_move_keys)

            # we visit the correct amount of nodes
            visit_counts = [child.visit_count for child in root.children.values()]
            self.assertEqual(sum(visit_counts), config.mcts.num_simulations)

            make_move(ctypes.byref(board), selected_move)

    # def test_mate_in_1(self):
    #     fen = "1k6/ppp5/8/8/8/8/8/4K2R w - - 0 1"
    #     winning_move = "h1h8"
    #     best_move = find_best_move(fen, 2000, self.real_net)
    #     self.assertEqual(best_move, winning_move)

    # def test_mate_in_1_black(self):
    #     fen = "qk6/8/8/8/8/8/PPP5/1K6 b - - 0 1"
    #     winning_move = "a8h1"
    #     best_move = find_best_move(fen, 2000, self.real_net)
    #     self.assertEqual(best_move, winning_move)


    # def test_mate_in_1_capture(self):
    #     fen = "r1bqk2r/ppp1bppp/8/8/6PN/5P2/PPPPP3/RNBQKB2 b - - 0 1"
    #     winning_move = "e7h4"
    #     best_move = find_best_move(fen, 2000, self.real_net) # these should not require this many simulations
    #     self.assertEqual(best_move, winning_move)

    # def test_mate_in_2(self):
    #     fen = "r1bqk2r/ppp1bppp/8/5N2/6P1/5P2/PPPPP3/RNBQKB2 b Qkq - 2 6"
    #     winning_move = "e7h4"
    #     best_move = find_best_move(fen, 2000, self.real_net)
    #     self.assertEqual(best_move, winning_move)

    # def test_mate_in_3(self):
    #     fen = "r1bqk2r/ppp1bppp/8/4B3/6P1/5P2/PPPPP3/3RKB1R b kq - 2 6"
    #     winning_move = "e7h4"
    #     best_move = find_best_move(fen, 2000, self.real_net)
    #     self.assertEqual(best_move, winning_move)

if __name__ == "__main__":
    unittest.main()
