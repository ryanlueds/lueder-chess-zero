#ifndef MASKS_AND_ATTACKS_H
#define MASKS_AND_ATTACKS_H

#include "../../../chess/attacks/static_attacks.h"
#include "../../../common/types.h"

extern const uint64_t bmagic[64];
extern const uint64_t rmagic[64];
extern const int bishop_mask_popcount[64];
extern const int rook_mask_popcount[64];

// PLAIN magic bitboard attack maps things for each square
// bishop attack mask has at most 512 moves
extern uint64_t bishop_attack_map[64][512];
// rook attack mask has at most 4096 moves (corners)
extern uint64_t rook_attack_map[64][4096];

// bitboards of relevant pieces and information
extern uint64_t pawn_attacks[2][64];
extern uint64_t knight_attacks[64];
extern uint64_t king_attacks[64];
extern uint64_t rook_masks[64];
extern uint64_t bishop_masks[64];

static inline uint64_t get_bishop_attacks(int square, uint64_t mask) {
    mask &= bishop_masks[square];
    mask *= bmagic[square];
    mask >>= 64 - bishop_mask_popcount[square];

    return bishop_attack_map[square][mask];
}

static inline uint64_t get_rook_attacks(int square, uint64_t mask) {
    mask &= rook_masks[square];
    mask *= rmagic[square];
    mask >>= 64 - rook_mask_popcount[square];

    return rook_attack_map[square][mask];
}

static inline uint64_t get_queen_attacks(int square, uint64_t mask) {
    return get_rook_attacks(square, mask) | get_bishop_attacks(square, mask);
}

void generate_bishop_attacks();
void generate_rook_attacks();
void generate_leaper_attacks();
void generate_slider_masks();
void init_all_attack_tables();

#endif
