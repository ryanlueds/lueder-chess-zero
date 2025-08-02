#include "move_generation.h"
#include <string.h>

// clang-format off
// castling rights update constants
const int castling_rights[64] = {
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14
};
// clang-format on

void generate_moves(const Board* board, Moves* move_list) {
    move_list->count = 0;

    int source_square, target_square;
    uint64_t bitboard, attacks;

    const int side = board->side;
    const int opponent_side = side ^ 1;
    const int king_square = bitscan_forward(board->bitboards[(side == white) ? K : k]);

    uint64_t pin_rays[64];
    const uint64_t pinned_pieces = get_pinned_pieces(board, side, pin_rays);
    const uint64_t checkers = is_in_check(board, side);

    // If in double check, only king moves are legal
    if (popcount(checkers) > 1) {
        int piece = (side == white) ? K : k;
        attacks = king_attacks[king_square] & ~board->occupancies[side];

        while (attacks) {
            target_square = bitscan_forward(attacks);

            uint64_t temp_occupancies = board->occupancies[both];
            pop_bit(&temp_occupancies, king_square);
            set_bit(&temp_occupancies, target_square);

            if (!is_square_attacked_through_king(board, target_square, opponent_side, temp_occupancies)) {
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list,
                         encode_move(king_square, target_square, piece, 0, is_capture, 0, 0, 0));
            }
            pop_bit(&attacks, target_square);
        }
        return;
    }

    uint64_t check_mask = ~0ULL;
    if (checkers) {
        int checker_square = bitscan_forward(checkers);
        check_mask = (1ULL << checker_square);

        int piece_type = -1;
        for (int piece = (opponent_side == white ? P : p); piece <= (opponent_side == white ? K : k); piece++) {
            if (get_bit(board->bitboards[piece], checker_square)) {
                piece_type = piece;
                break;
            }
        }

        if (piece_type == R || piece_type == r || piece_type == B || piece_type == b || piece_type == Q ||
            piece_type == q) {
            if (get_rook_attacks(king_square, board->occupancies[both]) & (1ULL << checker_square)) {
                check_mask |= (get_rook_attacks(king_square, board->occupancies[both]) &
                               get_rook_attacks(checker_square, board->occupancies[both]));
            } else {
                check_mask |= (get_bishop_attacks(king_square, board->occupancies[both]) &
                               get_bishop_attacks(checker_square, board->occupancies[both]));
            }
        }
    }

    if (side == white) {
        // white pawn moves
        int piece = P;
        bitboard = board->bitboards[piece];

        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;
            
            // pawn pushes

            target_square = source_square - 8;
            if (target_square >= a8 && !get_bit(board->occupancies[both], target_square)) {
                if ((1ULL << target_square) & check_mask & pin_ray) {
                    if (source_square >= a7 && source_square <= h7) {  // promotions
                        add_move(move_list, encode_move(source_square, target_square, piece, Q, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, R, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, B, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, N, 0, 0, 0, 0));
                    } else {
                        add_move(move_list, encode_move(source_square, target_square, piece, 0, 0, 0, 0, 0));
                    }
                }
                // double push
                if (source_square >= a2 && source_square <= h2) {
                    target_square = source_square - 16;
                    if (!get_bit(board->occupancies[both], target_square) &&
                        !get_bit(board->occupancies[both], source_square - 8)) {
                        if ((1ULL << target_square) & check_mask & pin_ray) {
                            add_move(move_list,
                                     encode_move(source_square, target_square, piece, 0, 0, 1, 0, 0));
                        }
                    }
                }
            }


            // pawn captures
            attacks = pawn_attacks[side][source_square] & board->occupancies[opponent_side];
            attacks &= check_mask & pin_ray;
            while (attacks) {
                target_square = bitscan_forward(attacks);
                if (source_square >= a7 && source_square <= h7) {  // promotions
                    add_move(move_list, encode_move(source_square, target_square, piece, Q, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, R, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, B, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, N, 1, 0, 0, 0));
                } else {
                    add_move(move_list, encode_move(source_square, target_square, piece, 0, 1, 0, 0, 0));
                }
                pop_bit(&attacks, target_square);
            }

            // En passant
            if (board->enpassant != no_sq) {
                uint64_t enpassant_attacks =
                    pawn_attacks[side][source_square] & (1ULL << board->enpassant) & pin_ray;
                uint64_t temp_check_mask = check_mask;

                // we are in check by a black pawn which we can en passant.
                if (board->side == white &&
                    (temp_check_mask & (1ULL << (board->enpassant + 8)) & board->bitboards[p])) {
                    set_bit(&temp_check_mask, board->enpassant);
                }

                enpassant_attacks &= temp_check_mask;

                if (enpassant_attacks) {
                    int capture_square = board->enpassant + 8;

                    // Simulate the board after en passant
                    uint64_t temp_occupancies = board->occupancies[both];
                    pop_bit(&temp_occupancies, source_square);
                    pop_bit(&temp_occupancies, capture_square);
                    set_bit(&temp_occupancies, board->enpassant);

                    if (!is_square_attacked_by_sliders(board, king_square, opponent_side, temp_occupancies)) {
                        add_move(move_list, encode_move(source_square, board->enpassant, piece, 0, 1, 0, 1, 0));
                    }
                }
            }
            
            pop_bit(&bitboard, source_square);
        }

        // white king moves
        piece = K;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;
            attacks = king_attacks[source_square] & ~board->occupancies[side];
            while (attacks) {
                target_square = bitscan_forward(attacks);

                uint64_t temp_occupancies = board->occupancies[both];
                pop_bit(&temp_occupancies, king_square);
                set_bit(&temp_occupancies, target_square);

                if (!is_square_attacked_through_king(board, target_square, opponent_side, temp_occupancies)) {
                    int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                    add_move(move_list, encode_move(king_square, target_square, piece, 0, is_capture, 0, 0, 0));
                }
                pop_bit(&attacks, target_square);
            }
            // Castling
            if (!checkers) {
                if ((board->castle & wk) && !get_bit(board->occupancies[both], f1) &&
                    !get_bit(board->occupancies[both], g1)) {
                    if (!is_square_attacked(board, e1, black) && !is_square_attacked(board, f1, black) &&
                        !is_square_attacked(board, g1, black)) {
                        add_move(move_list, encode_move(e1, g1, piece, 0, 0, 0, 0, 1));
                    }
                }
                if ((board->castle & wq) && !get_bit(board->occupancies[both], d1) &&
                    !get_bit(board->occupancies[both], c1) && !get_bit(board->occupancies[both], b1)) {
                    if (!is_square_attacked(board, e1, black) && !is_square_attacked(board, d1, black) &&
                        !is_square_attacked(board, c1, black)) {
                        add_move(move_list, encode_move(e1, c1, piece, 0, 0, 0, 0, 1));
                    }
                }
            }
            pop_bit(&bitboard, source_square);
        }

        // white knight moves
        piece = N;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = knight_attacks[source_square];
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // white bishop moves
        piece = B;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_bishop_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // white rook moves
        piece = R;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_rook_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // white queen moves
        piece = Q;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_queen_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }
    } else {
        // black pawn moves
        int piece = p;
        bitboard = board->bitboards[piece];

        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;
            // pawn pushes
            target_square = source_square + 8;
            if (target_square <= h1 && !get_bit(board->occupancies[both], target_square)) {
                if ((1ULL << target_square) & check_mask & pin_ray) {
                    if (source_square >= a2 && source_square <= h2) {  // Promotions
                        add_move(move_list, encode_move(source_square, target_square, piece, q, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, r, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, b, 0, 0, 0, 0));
                        add_move(move_list, encode_move(source_square, target_square, piece, n, 0, 0, 0, 0));
                    } else {
                        add_move(move_list, encode_move(source_square, target_square, piece, 0, 0, 0, 0, 0));
                    }
                }
                // Double push
                if (source_square >= a7 && source_square <= h7) {
                    target_square = source_square + 16;
                    if (!get_bit(board->occupancies[both], target_square) &&
                        !get_bit(board->occupancies[both], source_square + 8)) {
                        if ((1ULL << target_square) & check_mask & pin_ray) {
                            add_move(move_list,
                                     encode_move(source_square, target_square, piece, 0, 0, 1, 0, 0));
                        }
                    }
                }
            }

            // Pawn captures
            attacks = pawn_attacks[side][source_square] & board->occupancies[opponent_side];
            attacks &= check_mask & pin_ray;
            while (attacks) {
                target_square = bitscan_forward(attacks);
                if (source_square >= a2 && source_square <= h2) {
                    add_move(move_list, encode_move(source_square, target_square, piece, q, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, r, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, b, 1, 0, 0, 0));
                    add_move(move_list, encode_move(source_square, target_square, piece, n, 1, 0, 0, 0));
                } else {
                    add_move(move_list, encode_move(source_square, target_square, piece, 0, 1, 0, 0, 0));
                }
                pop_bit(&attacks, target_square);
            }

            // En passant
            if (board->enpassant != no_sq) {
                uint64_t enpassant_attacks =
                    pawn_attacks[side][source_square] & (1ULL << board->enpassant) & pin_ray;
                uint64_t temp_check_mask = check_mask;

                // we are in check by a white pawn which we can en passant.
                if (board->side == black &&
                         (temp_check_mask & (1ULL << (board->enpassant - 8)) & board->bitboards[P])) {
                    set_bit(&temp_check_mask, board->enpassant);
                }

                enpassant_attacks &= temp_check_mask;

                if (enpassant_attacks) {
                    int capture_square = board->enpassant - 8;

                    // Simulate the board after en passant
                    uint64_t temp_occupancies = board->occupancies[both];
                    pop_bit(&temp_occupancies, source_square);
                    pop_bit(&temp_occupancies, capture_square);
                    set_bit(&temp_occupancies, board->enpassant);

                    if (!is_square_attacked_by_sliders(board, king_square, opponent_side, temp_occupancies)) {
                        add_move(move_list, encode_move(source_square, board->enpassant, piece, 0, 1, 0, 1, 0));
                    }
                }
            }
            pop_bit(&bitboard, source_square);
        }

        // black king moves
        piece = k;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;
            attacks = king_attacks[source_square] & ~board->occupancies[side];
            while (attacks) {
                target_square = bitscan_forward(attacks);

                uint64_t temp_occupancies = board->occupancies[both];
                pop_bit(&temp_occupancies, king_square);
                set_bit(&temp_occupancies, target_square);

                if (!is_square_attacked_through_king(board, target_square, opponent_side, temp_occupancies)) {
                    int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                    add_move(move_list, encode_move(king_square, target_square, piece, 0, is_capture, 0, 0, 0));
                }
                pop_bit(&attacks, target_square);
            }
            // Castling
            if (!checkers) {
                if ((board->castle & bk) && !get_bit(board->occupancies[both], f8) &&
                    !get_bit(board->occupancies[both], g8)) {
                    if (!is_square_attacked(board, e8, white) && !is_square_attacked(board, f8, white) &&
                        !is_square_attacked(board, g8, white)) {
                        add_move(move_list, encode_move(e8, g8, piece, 0, 0, 0, 0, 1));
                    }
                }
                if ((board->castle & bq) && !get_bit(board->occupancies[both], d8) &&
                    !get_bit(board->occupancies[both], c8) && !get_bit(board->occupancies[both], b8)) {
                    if (!is_square_attacked(board, e8, white) && !is_square_attacked(board, d8, white) &&
                        !is_square_attacked(board, c8, white)) {
                        add_move(move_list, encode_move(e8, c8, piece, 0, 0, 0, 0, 1));
                    }
                }
            }
            pop_bit(&bitboard, source_square);
        }

        // black knight moves
        piece = n;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = knight_attacks[source_square];
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // black bishop moves
        piece = b;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_bishop_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // black rook moves
        piece = r;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_rook_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }

        // black queen moves
        piece = q;
        bitboard = board->bitboards[piece];
        while (bitboard) {
            source_square = bitscan_forward(bitboard);
            uint64_t pin_ray = get_bit(pinned_pieces, source_square) ? pin_rays[source_square] : ~0ULL;

            attacks = get_queen_attacks(source_square, board->occupancies[both]);
            attacks &= ~board->occupancies[side];
            attacks &= check_mask & pin_ray;

            while (attacks) {
                target_square = bitscan_forward(attacks);
                int is_capture = get_bit(board->occupancies[opponent_side], target_square) ? 1 : 0;
                add_move(move_list, encode_move(source_square, target_square, piece, 0, is_capture, 0, 0, 0));
                pop_bit(&attacks, target_square);
            }
            pop_bit(&bitboard, source_square);
        }
    }

    // print_move_list(move_list);
}

int make_move(Board* board, int move) {
    int source_square = get_move_source(move);
    int target_square = get_move_target(move);
    int piece = get_move_piece(move);
    int promoted_piece = get_move_promoted(move);
    int capture = get_move_capture(move);
    int double_push = get_move_double(move);
    int enpass = get_move_enpassant(move);
    int castling = get_move_castling(move);

    int side = board->side;

    uint64_t from_to_mask = (1ULL << source_square) | (1ULL << target_square);

    // fast path for quiet moves
    if (!(capture | promoted_piece | enpass | castling | double_push)) {
        board->bitboards[piece] ^= from_to_mask;
        board->occupancies[side] ^= from_to_mask;
        board->occupancies[both] ^= from_to_mask;
        board->castle &= castling_rights[source_square];
        board->castle &= castling_rights[target_square];
        board->enpassant = no_sq;
        board->side ^= 1;
        return 0;
    }

    // move
    board->bitboards[piece] &= ~(1ULL << source_square);
    board->bitboards[piece] |= (1ULL << target_square);

    // handle capture
    int captured_piece = 0;
    if (capture && !enpass) {
        int cap_piece_start = (side == white) ? p : P;
        int cap_piece_end = (side == white) ? k : K;
        for (int bb = cap_piece_start; bb <= cap_piece_end; bb++) {
            if (get_bit(board->bitboards[bb], target_square)) {
                captured_piece = bb;
                board->bitboards[bb] &= ~(1ULL << target_square);
                break;
            }
        }
    }

    // handle promotion
    if (promoted_piece) {
        int pawn = (side == white) ? P : p;
        board->bitboards[pawn] &= ~(1ULL << target_square);
        board->bitboards[promoted_piece] |= (1ULL << target_square);
    }

    // handle en passant
    if (enpass) {
        int ep_capture_sq = (side == white) ? target_square + 8 : target_square - 8;
        int ep_pawn = (side == white) ? p : P;
        board->bitboards[ep_pawn] &= ~(1ULL << ep_capture_sq);
        // TODO:idk
        captured_piece = ep_pawn;
    }

    // no more en passant
    board->enpassant = no_sq;

    // double pawn push
    if (double_push) {
        board->enpassant = (side == white) ? target_square + 8 : target_square - 8;
    }

    // caslting rights
    if (castling) {
        if (target_square == g1) {  // white king-side
            board->bitboards[R] ^= (1ULL << h1) | (1ULL << f1);
        } else if (target_square == c1) {  // white queen-side
            board->bitboards[R] ^= (1ULL << a1) | (1ULL << d1);
        } else if (target_square == g8) {  // black king-side
            board->bitboards[r] ^= (1ULL << h8) | (1ULL << f8);
        } else if (target_square == c8) {  // black queen-side
            board->bitboards[r] ^= (1ULL << a8) | (1ULL << d8);
        }
    }

    // update castling rights
    board->castle &= castling_rights[source_square];
    board->castle &= castling_rights[target_square];

    // update occupancies
    board->occupancies[white] = 0ULL;
    board->occupancies[white] |= board->bitboards[P];
    board->occupancies[white] |= board->bitboards[B];
    board->occupancies[white] |= board->bitboards[N];
    board->occupancies[white] |= board->bitboards[K];
    board->occupancies[white] |= board->bitboards[Q];
    board->occupancies[white] |= board->bitboards[R];
    board->occupancies[black] = 0ULL;
    board->occupancies[black] |= board->bitboards[p];
    board->occupancies[black] |= board->bitboards[b];
    board->occupancies[black] |= board->bitboards[n];
    board->occupancies[black] |= board->bitboards[k];
    board->occupancies[black] |= board->bitboards[q];
    board->occupancies[black] |= board->bitboards[r];
    board->occupancies[both] = board->occupancies[white] | board->occupancies[black];

    board->side ^= 1;

    return captured_piece;
}

void undo_move(Board* board, MoveData* data) {
    int move = data->move;
    int source = get_move_source(move);
    int target = get_move_target(move);
    int piece = get_move_piece(move);
    int promoted = get_move_promoted(move);
    int capture = get_move_capture(move);
    int enpass = get_move_enpassant(move);
    int castling = get_move_castling(move);

    board->side ^= 1;
    board->castle = data->castling_rights;
    board->enpassant = data->enpassant_square;
    memcpy(board->occupancies, data->occupancies, sizeof(data->occupancies));

    // Undo castling
    if (castling) {
        switch (target) {
            case g1:
                pop_bit(&board->bitboards[R], f1);
                set_bit(&board->bitboards[R], h1);
                break;
            case c1:
                pop_bit(&board->bitboards[R], d1);
                set_bit(&board->bitboards[R], a1);
                break;
            case g8:
                pop_bit(&board->bitboards[r], f8);
                set_bit(&board->bitboards[r], h8);
                break;
            case c8:
                pop_bit(&board->bitboards[r], d8);
                set_bit(&board->bitboards[r], a8);
                break;
        }
    }

    // Undo promotion
    if (promoted) {
        pop_bit(&board->bitboards[promoted], target);
        set_bit(&board->bitboards[(board->side == white) ? P : p], source);
    } else {
        pop_bit(&board->bitboards[piece], target);
        set_bit(&board->bitboards[piece], source);
    }

    // Undo capture
    if (capture && !enpass) {
        set_bit(&board->bitboards[data->captured_piece], target);
    }

    // Undo en passant capture
    if (enpass) {
        if (board->side == white) {
            set_bit(&board->bitboards[p], target + 8);
        } else {
            set_bit(&board->bitboards[P], target - 8);
        }
    }
}

void print_attacked_squares(const Board* board, int side) {
    printf("\n");

    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;

            if (!file)
                printf("  %d ", 8 - rank);

            printf(" %d", is_square_attacked(board, square, side) ? 1 : 0);
        }

        printf("\n");
    }

    printf("\n     a b c d e f g h\n\n");
}

void print_move_list(Moves* move_list) {
    if (!move_list->count) {
        printf("\n     No move in the move list!\n");
        return;
    }

    printf("\n     move    piece     capture   double    enpass    castling\n\n");

    for (int move_count = 0; move_count < move_list->count; move_count++) {
        int move = move_list->moves[move_count];

        printf("     %s%s%c   %s         %d         %d         %d         %d\n",
               square_to_coordinates[get_move_source(move)], square_to_coordinates[get_move_target(move)],
               get_move_promoted(move) ? promoted_pieces[get_move_promoted(move)] : ' ',
               unicode_pieces[get_move_piece(move)], get_move_capture(move) ? 1 : 0, get_move_double(move) ? 1 : 0,
               get_move_enpassant(move) ? 1 : 0, get_move_castling(move) ? 1 : 0);
    }

    printf("\n\n     Total number of moves: %d\n\n", move_list->count);
}

void print_move(int move) {
    if (get_move_promoted(move)) {
        printf("%s%s%c\n", square_to_coordinates[get_move_source(move)], square_to_coordinates[get_move_target(move)],
               promoted_pieces[get_move_promoted(move)]);
    } else {
        printf("%s%s\n", square_to_coordinates[get_move_source(move)], square_to_coordinates[get_move_target(move)]);
    }
}

// for my python!
void copy_board(Board* original, Board* copy) {
    memcpy(copy, original, sizeof(Board));
}

void take_back(Board* original, Board* copy) {
    memcpy(original, copy, sizeof(Board));
}

uint64_t is_in_check_wrapper(const Board* board, int side) {
    return is_in_check(board, side);
}
