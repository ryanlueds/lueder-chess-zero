import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wrapper import Board, Moves, init_all_attack_tables, parse_fen, generate_moves, make_move

class TestWrapper(unittest.TestCase):
    def setUp(self):
        init_all_attack_tables()
        self.board = Board()
        parse_fen(self.board, b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_fen_parsing(self):
        self.assertEqual(self.board.side, 0)
        self.assertEqual(self.board.castle, 15)
        self.assertEqual(self.board.enpassant, 64) # enpassant is on no_sq

    def test_move_generation(self):
        moves = Moves()
        generate_moves(self.board, moves)
        self.assertEqual(moves.count, 20)

    def test_make_move(self):
        moves = Moves()
        generate_moves(self.board, moves)
        move = moves.moves[0]
        make_move(self.board, move)
        self.assertEqual(self.board.side, 1)

if __name__ == '__main__':
    unittest.main()
