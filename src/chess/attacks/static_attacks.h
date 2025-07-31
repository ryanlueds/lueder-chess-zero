#ifndef STATIC_ATTACKS_H
#define STATIC_ATTACKS_H

#include "../../common/bitboard_utils.h"
#include "../../common/types.h"

// not A file and not H file
static const uint64_t not_a_file = 18374403900871474942ULL;
static const uint64_t not_h_file = 9187201950435737471ULL;
static const uint64_t not_gh_file = 4557430888798830399ULL;
static const uint64_t not_ab_file = 18229723555195321596ULL;

uint64_t get_rook_mask(int square);
uint64_t get_bishop_mask(int square);
uint64_t get_rook_attack(int square, uint64_t block);
uint64_t get_bishop_attack(int square, uint64_t block);
uint64_t get_pawn_attack(int side, int square);
uint64_t get_knight_attack(int square);
uint64_t get_king_attack(int square);

#endif
