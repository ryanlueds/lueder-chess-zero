#include "static_attacks.h"

uint64_t get_rook_mask(int square) {
    uint64_t attacks = 0ULL;

    int start_rank = square / 8;
    int start_file = square % 8;

    int rank, file;

    // N Direction (not including border)
    for (rank = start_rank - 1; rank > 0; rank--) {
        set_bit(&attacks, (8 * rank + start_file));
    }

    // S Direcion (not including border)
    for (rank = start_rank + 1; rank < 7; rank++) {
        set_bit(&attacks, (8 * rank + start_file));
    }

    // E Direction (not including border)
    for (file = start_file + 1; file < 7; file++) {
        set_bit(&attacks, (8 * start_rank + file));
    }

    // W Direction (not including border)
    for (file = start_file - 1; file > 0; file--) {
        set_bit(&attacks, (8 * start_rank + file));
    }

    return attacks;
}

uint64_t get_bishop_mask(int square) {
    uint64_t attacks = 0ULL;

    int start_rank = square / 8;
    int start_file = square % 8;

    int rank, file;

    // SE Diagonal (not including border)
    for (rank = start_rank + 1, file = start_file + 1; rank < 7 && file < 7; rank++, file++) {
        set_bit(&attacks, (rank * 8 + file));
    }
    // NW Diagonal (not including border)
    for (rank = start_rank - 1, file = start_file - 1; rank > 0 && file > 0; rank--, file--) {
        set_bit(&attacks, (rank * 8 + file));
    }

    // NE Diagonal (not including border)
    for (rank = start_rank - 1, file = start_file + 1; rank > 0 && file < 7; rank--, file++) {
        set_bit(&attacks, (rank * 8 + file));
    }

    // SW Diagonal (not including border)
    for (rank = start_rank + 1, file = start_file - 1; rank < 7 && file > 0; rank++, file--) {
        set_bit(&attacks, (rank * 8 + file));
    }

    return attacks;
}

uint64_t get_bishop_attack(int square, uint64_t block) {
    uint64_t attacks = 0ULL;

    int start_rank = square / 8;
    int start_file = square % 8;

    int rank, file;

    // SE Diagonal (not including border)
    for (rank = start_rank + 1, file = start_file + 1; rank <= 7 && file <= 7; rank++, file++) {
        set_bit(&attacks, (rank * 8 + file));
        if (get_bit(attacks, (rank * 8 + file)) & block) {
            break;
        }
    }
    // NW Diagonal (not including border)
    for (rank = start_rank - 1, file = start_file - 1; rank >= 0 && file >= 0; rank--, file--) {
        set_bit(&attacks, (rank * 8 + file));
        if (get_bit(attacks, (rank * 8 + file)) & block) {
            break;
        }
    }

    // NE Diagonal (not including border)
    for (rank = start_rank - 1, file = start_file + 1; rank >= 0 && file <= 7; rank--, file++) {
        set_bit(&attacks, (rank * 8 + file));
        if (get_bit(attacks, (rank * 8 + file)) & block) {
            break;
        }
    }

    // SW Diagonal (not including border)
    for (rank = start_rank + 1, file = start_file - 1; rank <= 7 && file >= 0; rank++, file--) {
        set_bit(&attacks, (rank * 8 + file));
        if (get_bit(attacks, (rank * 8 + file)) & block) {
            break;
        }
    }

    return attacks;
}

uint64_t get_rook_attack(int square, uint64_t block) {
    uint64_t attacks = 0ULL;

    int start_rank = square / 8;
    int start_file = square % 8;

    int rank, file;

    // N Direction (not including border)
    for (rank = start_rank - 1; rank >= 0; rank--) {
        set_bit(&attacks, (8 * rank + start_file));
        if (get_bit(attacks, (rank * 8 + start_file)) & block) {
            break;
        }
    }

    // S Direcion (not including border)
    for (rank = start_rank + 1; rank <= 7; rank++) {
        set_bit(&attacks, (8 * rank + start_file));
        if (get_bit(attacks, (rank * 8 + start_file)) & block) {
            break;
        }
    }

    // E Direction (not including border)
    for (file = start_file + 1; file <= 7; file++) {
        set_bit(&attacks, (8 * start_rank + file));
        if (get_bit(attacks, (start_rank * 8 + file)) & block) {
            break;
        }
    }

    // W Direction (not including border)
    for (file = start_file - 1; file >= 0; file--) {
        set_bit(&attacks, (8 * start_rank + file));
        if (get_bit(attacks, (start_rank * 8 + file)) & block) {
            break;
        }
    }

    return attacks;
}

uint64_t get_pawn_attack(int side, int square) {
    uint64_t attacks = 0ULL;
    uint64_t bb = 0ULL;
    set_bit(&bb, square);

    if (side == white) {
        // top right
        attacks |= ((bb & not_h_file) >> 7);
        // top left
        attacks |= ((bb & not_a_file) >> 9);
    } else {
        // bot left
        attacks |= ((bb & not_a_file) << 7);
        // bot right
        attacks |= ((bb & not_h_file) << 9);
    }

    return attacks;
}

uint64_t get_knight_attack(int square) {
    uint64_t attacks = 0ULL;
    uint64_t bb = 0ULL;
    set_bit(&bb, square);

    // top top left
    attacks |= ((bb & not_a_file) >> 17);
    // top top right
    attacks |= ((bb & not_h_file) >> 15);
    // top right
    attacks |= ((bb & not_gh_file) >> 6);
    // top left
    attacks |= ((bb & not_ab_file) >> 10);
    // bot bot right
    attacks |= ((bb & not_h_file) << 17);
    // bot bot left
    attacks |= ((bb & not_a_file) << 15);
    // bot left
    attacks |= ((bb & not_ab_file) << 6);
    // bot right
    attacks |= ((bb & not_gh_file) << 10);

    return attacks;
}

uint64_t get_king_attack(int square) {
    uint64_t attacks = 0ULL;
    uint64_t bb = 0ULL;
    set_bit(&bb, square);

    // top left
    attacks |= ((bb & not_a_file) >> 9);
    // top
    attacks |= (bb >> 8);
    // top right
    attacks |= ((bb & not_h_file) >> 7);
    // left
    attacks |= ((bb & not_a_file) >> 1);
    // bot right
    attacks |= ((bb & not_h_file) << 9);
    // bot
    attacks |= (bb << 8);
    // bot left
    attacks |= ((bb & not_a_file) << 7);
    // right
    attacks |= ((bb & not_h_file) << 1);

    return attacks;
}
