#include "test_suite.h"
#include <stdbool.h>
#include <stdio.h>
#include "../common/bitboard_utils.h"
#include "../framework/chess/board/board.h"
#include "../framework/chess/movegeneration/move_generation.h"
#include "test_macros.h"

// Forward declarations
void test_is_in_check();
void test_get_pinned_pieces();
void test_move_generation();

void test_is_in_check() {
    printf("  Testing is_in_check...\n");
    Board board;
    uint64_t checkers;

    parse_fen(&board, "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1");
    checkers = is_in_check(&board, black);
    EXPECT_EQ(popcount(checkers), 1, "Rook check popcount");
    EXPECT_TRUE(checkers & (1ULL << e2), "Rook check square");

    parse_fen(&board, "4k3/8/8/8/8/5n2/8/4K3 w - - 0 1");
    checkers = is_in_check(&board, white);
    EXPECT_EQ(popcount(checkers), 1, "Knight check popcount");
    EXPECT_TRUE(checkers & (1ULL << f3), "Knight check square");

    parse_fen(&board, "4k3/8/8/8/8/2b5/8/4K3 w - - 0 1");
    checkers = is_in_check(&board, white);
    EXPECT_EQ(popcount(checkers), 1, "Bishop check popcount");
    EXPECT_TRUE(checkers & (1ULL << c3), "Bishop check square");

    parse_fen(&board, "4k3/8/8/8/8/2b5/8/3rK3 w - - 0 1");
    checkers = is_in_check(&board, white);
    EXPECT_EQ(popcount(checkers), 2, "Double check popcount");
    EXPECT_TRUE(checkers & (1ULL << c3), "Double check bishop");
    EXPECT_TRUE(checkers & (1ULL << d1), "Double check rook");

    parse_fen(&board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    checkers = is_in_check(&board, white);
    EXPECT_EQ(popcount(checkers), 0, "No check (white)");
    checkers = is_in_check(&board, black);
    EXPECT_EQ(popcount(checkers), 0, "No check (black)");
}

void test_get_pinned_pieces() {
    printf("  Testing get_pinned_pieces...\n");
    Board board;
    uint64_t pinned;
    uint64_t pin_rays[64];

    parse_fen(&board, "4k3/8/8/8/8/4b3/4R3/4K3 w - - 0 1");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, black, pin_rays);
    EXPECT_EQ(popcount(pinned), 1, "Rook pin popcount");
    EXPECT_TRUE(pinned & (1ULL << e3), "Rook pin piece");
    EXPECT_NONZERO(pin_rays[e3], "Rook pin ray");

    parse_fen(&board, "7k/8/8/8/3n4/8/8/B3K3 b - - 0 1");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, white, pin_rays);
    EXPECT_EQ(popcount(pinned), 0, "No white pins");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, black, pin_rays);
    EXPECT_EQ(popcount(pinned), 1, "Bishop pin popcount");
    EXPECT_TRUE(pinned & (1ULL << d4), "Bishop pin piece");
    EXPECT_NONZERO(pin_rays[d4], "Bishop pin ray");

    parse_fen(&board, "7k/8/8/4K3/8/2N5/8/q7 w - - 0 1");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, white, pin_rays);
    EXPECT_EQ(popcount(pinned), 1, "Queen pin popcount");
    EXPECT_TRUE(pinned & (1ULL << c3), "Queen pin piece");
    EXPECT_NONZERO(pin_rays[c3], "Queen pin ray");

    parse_fen(&board, "7Q/8/8/8/4K2n/8/8/R2n3k b - - 0 1");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, black, pin_rays);
    EXPECT_EQ(popcount(pinned), 2, "Multiple pins count");
    EXPECT_TRUE(pinned & (1ULL << d1), "Multiple pin d1");
    EXPECT_TRUE(pinned & (1ULL << h4), "Multiple pin h4");
    EXPECT_NONZERO(pin_rays[d1], "Pin ray for d1");
    EXPECT_NONZERO(pin_rays[h4], "Pin ray for h4");

    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, white, pin_rays);
    EXPECT_EQ(popcount(pinned), 0, "No white pins in multiple pin case");

    parse_fen(&board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, white, pin_rays);
    EXPECT_EQ(popcount(pinned), 0, "No pin (white)");
    memset(pin_rays, 0, 64 * sizeof(uint64_t));
    pinned = get_pinned_pieces(&board, black, pin_rays);
    EXPECT_EQ(popcount(pinned), 0, "No pin (black)");
}

void test_move_generation() {
    printf("  Testing generate_moves...\n");
    Board board;
    Moves move_list[1];

    parse_fen(&board, "4k3/3p4/8/1B6/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "Pinned pawn (from bishop) move count");

    parse_fen(&board, "4k3/3p4/8/1Q6/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "Pinned pawn (from queen) move count");

    parse_fen(&board, "4k3/3p4/8/1B1N4/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "Pinned pawn (from bishop) and avoiding check move count");

    parse_fen(&board, "4k3/3p4/8/3N4/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "No double pawn push (1)");
    parse_fen(&board, "4k3/3p4/3B4/8/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 2, "No double pawn push (2)");

    parse_fen(&board, "4k3/8/2p5/1PPP4/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 7, "Two pawn caputes (1)");
    parse_fen(&board, "4k3/1p6/2p5/1PPP4/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 8, "Two pawn caputes (2)");

    parse_fen(&board, "4k3/4p3/5B2/8/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 7, "Pawn capture on starting rank");

    parse_fen(&board, "4k3/3p4/8/8/8/4R3/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "King in check by rook");

    parse_fen(&board, "4k3/3p4/8/7B/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "King in check by bishop");

    parse_fen(&board, "4k3/3p4/8/8/7B/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "King legal moves avoiding attack");

    parse_fen(&board, "4k2r/7p/8/8/8/8/8/4K3 b KQk - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 10, "King can castle kingside");

    parse_fen(&board, "4k2r/7p/8/3B4/8/8/8/4K3 b k - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 8, "King cannot castle kingside");

    parse_fen(&board, "4k1nr/7p/8/8/8/8/8/4K3 b KQk - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 10, "Cannot castle kingside");

    parse_fen(&board, "4k3/4r3/8/8/8/8/4R3/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 9, "Rook must move along pin");

    parse_fen(&board, "4k3/4r3/8/8/8/4B3/4R3/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 15, "Black rook can capture something");

    parse_fen(&board, "4kn2/8/6B1/8/8/8/8/4K3 b - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "Black knight can stop check");

    parse_fen(&board, "rnbqkbnr/ppppppp1/7p/1B6/8/4P3/PPPP1PPP/RNBQK1NR b KQkq - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 16, "Black pawn pinned in starting (1)");
    parse_fen(&board, "rnbqkbnr/ppppppp1/8/1B5p/8/4P3/PPPP1PPP/RNBQK1NR b KQkq - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 18, "Black pawn pinned in starting (2)");

    parse_fen(&board, "rnb1kbnr/pp1ppppp/2p5/q7/8/3P4/PPPKPPPP/RNBQ1BNR w kq - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 4, "King cannot walk along ray (1)");
    parse_fen(&board, "r6k/K7/8/8/8/8/8/8 w - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "King cannot walk along ray (2)");
    parse_fen(&board, "r6k/8/K7/8/8/8/8/8 w - - 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "King cannot walk along ray (3)");

    parse_fen(&board, "rnQq1k1r/pp3ppp/2p5/8/1bB5/8/PPPKNnPP/RNBQ3R w - - 0 8");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 1, "King cannot walk along ray in double check(4)");

    parse_fen(&board, "8/8/3p4/1Pp3kr/1K3R2/8/4P1P1/8 w - c6 0 3");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 7, "En passant to get king out of check");

    parse_fen(&board, "6bk/8/8/2pP4/2K5/8/8/8 w - c6 0 1");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 5, "Cannot en passant if pinned");

    parse_fen(&board, "3q4/8/3p4/KPp3kr/5R2/8/4P1P1/8 w - c6 0 3");
    generate_moves(&board, move_list);
    EXPECT_EQ(move_list->count, 3, "Cannot en passant if doesnt resolve check");
}
