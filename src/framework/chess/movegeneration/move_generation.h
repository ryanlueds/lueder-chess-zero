#ifndef MOVE_GENERATION_H
#define MOVE_GENERATION_H

#include "../../../common/bitboard_utils.h"
#include "../../../common/constants.h"
#include "../../../common/move_encoding.h"
#include "../../../common/types.h"
#include "../board/board.h"
#include "masks_and_attacks.h"

void copy_board(Board* original, Board* copy);
void take_back(Board* original, Board* copy);

enum { all_moves, only_captures };

extern const int castling_rights[64];

typedef struct {
    int moves[256];
    int count;
} Moves;

typedef struct {
    int move;
    int captured_piece;
    int castling_rights;
    int enpassant_square;
    uint64_t occupancies[3];
} MoveData;

void print_attacked_squares(const Board* board, int side);
void print_move_list(Moves* move_list);
void print_move(int move);
void generate_moves(const Board* board, Moves* move_list);
int make_move(Board* board, int move);
void undo_move(Board* board, MoveData* data);

// add move to the move list
static inline void add_move(Moves* move_list, int move) {
    move_list->moves[move_list->count] = move;
    move_list->count++;
}

// is square current given attacked by the current given side
static inline int is_square_attacked(const Board* board, int square, int side) {
    // white pawns
    if ((side == white) && (pawn_attacks[black][square] & board->bitboards[P]))
        return 1;
    // black pawns
    if ((side == black) && (pawn_attacks[white][square] & board->bitboards[p]))
        return 1;
    // knights
    if (knight_attacks[square] & ((side == white) ? board->bitboards[N] : board->bitboards[n]))
        return 1;
    // bishops
    if (get_bishop_attacks(square, board->occupancies[both]) &
        ((side == white) ? board->bitboards[B] : board->bitboards[b]))
        return 1;
    // rooks
    if (get_rook_attacks(square, board->occupancies[both]) &
        ((side == white) ? board->bitboards[R] : board->bitboards[r]))
        return 1;
    // queens
    if (get_queen_attacks(square, board->occupancies[both]) &
        ((side == white) ? board->bitboards[Q] : board->bitboards[q]))
        return 1;
    // kings
    if (king_attacks[square] & ((side == white) ? board->bitboards[K] : board->bitboards[k]))
        return 1;
    return 0;
}

// is square current given attacked by the current given side
static inline int is_square_attacked_through_king(const Board* board, int square, int side, uint64_t temp_occupancy) {
    // white pawns
    if ((side == white) && (pawn_attacks[black][square] & board->bitboards[P]))
        return 1;
    // black pawns
    if ((side == black) && (pawn_attacks[white][square] & board->bitboards[p]))
        return 1;
    // knights
    if (knight_attacks[square] & ((side == white) ? board->bitboards[N] : board->bitboards[n]))
        return 1;
    // bishops
    if (get_bishop_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[B] : board->bitboards[b]))
        return 1;
    // rooks
    if (get_rook_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[R] : board->bitboards[r]))
        return 1;
    // queens
    if (get_queen_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[Q] : board->bitboards[q]))
        return 1;
    return 0;
}

// is square current given attacked by the current given side
static inline int is_square_attacked_by_sliders(const Board* board, int square, int side, uint64_t temp_occupancy) {
    // bishops
    if (get_bishop_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[B] : board->bitboards[b]))
        return 1;
    // rooks
    if (get_rook_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[R] : board->bitboards[r]))
        return 1;
    // queens
    if (get_queen_attacks(square, temp_occupancy) & ((side == white) ? board->bitboards[Q] : board->bitboards[q]))
        return 1;
    return 0;
}

// is king in check
static inline uint64_t is_in_check(const Board* board, int side) {
    uint64_t checkers = 0;
    int king_square = bitscan_forward((side == white) ? board->bitboards[K] : board->bitboards[k]);

    // white king is in check from black pieces
    if (side == white) {
        checkers |= (pawn_attacks[white][king_square] & board->bitboards[p]);
        checkers |= (knight_attacks[king_square] & board->bitboards[n]);
        checkers |= (get_bishop_attacks(king_square, board->occupancies[both]) & board->bitboards[b]);
        checkers |= (get_rook_attacks(king_square, board->occupancies[both]) & board->bitboards[r]);
        checkers |= (get_queen_attacks(king_square, board->occupancies[both]) & board->bitboards[q]);
    }
    // black king is in check from white pieces
    else {
        checkers |= (pawn_attacks[black][king_square] & board->bitboards[P]);
        checkers |= (knight_attacks[king_square] & board->bitboards[N]);
        checkers |= (get_bishop_attacks(king_square, board->occupancies[both]) & board->bitboards[B]);
        checkers |= (get_rook_attacks(king_square, board->occupancies[both]) & board->bitboards[R]);
        checkers |= (get_queen_attacks(king_square, board->occupancies[both]) & board->bitboards[Q]);
    }

    return checkers;
}

static inline uint64_t get_pinned_pieces(const Board* board, int side, uint64_t pin_rays[64]) {
    uint64_t pinned = 0ULL;

    int king_square = bitscan_forward((side == white) ? board->bitboards[K] : board->bitboards[k]);

    uint64_t opponent_rooks_queens =
        (side == white) ? (board->bitboards[r] | board->bitboards[q]) : (board->bitboards[R] | board->bitboards[Q]);
    uint64_t opponent_bishops_queens =
        (side == white) ? (board->bitboards[b] | board->bitboards[q]) : (board->bitboards[B] | board->bitboards[Q]);

    // Rooks and Queens
    uint64_t pinners = get_rook_attacks(king_square, 0) & opponent_rooks_queens;
    while (pinners) {
        int pinner_square = bitscan_forward(pinners);
        uint64_t line_between =
            get_rook_attacks(king_square, 1ULL << pinner_square) & get_rook_attacks(pinner_square, 1ULL << king_square);
        uint64_t blockers = line_between & board->occupancies[both];

        if (popcount(blockers) == 1) {
            if (blockers & board->occupancies[side]) {
                pinned |= blockers;
                pin_rays[bitscan_forward(blockers)] = line_between | (1ULL << pinner_square) | (1ULL << king_square);
            }
        }
        pop_bit(&pinners, pinner_square);
    }

    // Bishops and Queens
    pinners = get_bishop_attacks(king_square, 0) & opponent_bishops_queens;
    while (pinners) {
        int pinner_square = bitscan_forward(pinners);
        uint64_t line_between = get_bishop_attacks(king_square, 1ULL << pinner_square) &
                                get_bishop_attacks(pinner_square, 1ULL << king_square);
        uint64_t blockers = line_between & board->occupancies[both];

        if (popcount(blockers) == 1) {
            if (blockers & board->occupancies[side]) {
                pinned |= blockers;
                pin_rays[bitscan_forward(blockers)] = line_between | (1ULL << pinner_square) | (1ULL << king_square);
            }
        }
        pop_bit(&pinners, pinner_square);
    }

    return pinned;
}

static inline int find_captured_piece_fast(Board* board, int move) {
    if (!get_move_capture(move))
        return 0;

    int target = get_move_target(move);
    if (get_move_enpassant(move))
        return 0;  // undo_move handles en passant

    uint64_t target_bit = 1ULL << target;
    int opponent = board->side ^ 1;

    // Check pieces in order of likelihood (pawns most common captures)
    if (opponent == white) {
        if (board->bitboards[P] & target_bit)
            return P;
        if (board->bitboards[N] & target_bit)
            return N;
        if (board->bitboards[B] & target_bit)
            return B;
        if (board->bitboards[R] & target_bit)
            return R;
        if (board->bitboards[Q] & target_bit)
            return Q;
        return K;  // Must be king if we got here
    } else {
        if (board->bitboards[p] & target_bit)
            return p;
        if (board->bitboards[n] & target_bit)
            return n;
        if (board->bitboards[b] & target_bit)
            return b;
        if (board->bitboards[r] & target_bit)
            return r;
        if (board->bitboards[q] & target_bit)
            return q;
        return k;  // Must be king if we got here
    }
}

static inline void undo_move_fast(Board* board,
                                  int move,
                                  int old_castle,
                                  int old_enpassant,
                                  int captured_piece,
                                  uint64_t* old_occupancies) {
    MoveData data;
    data.move = move;
    data.captured_piece = captured_piece;
    data.castling_rights = old_castle;
    data.enpassant_square = old_enpassant;
    data.occupancies[0] = old_occupancies[0];
    data.occupancies[1] = old_occupancies[1];
    data.occupancies[2] = old_occupancies[2];

    undo_move(board, &data);
}

uint64_t is_in_check_wrapper(const Board* board, int side);

#endif
