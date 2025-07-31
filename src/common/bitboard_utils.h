#ifndef BITBOARD_UTILS_H
#define BITBOARD_UTILS_H

#include "types.h"

// get/set/pop
static inline uint64_t get_bit(uint64_t bitboard, int square) {
    return bitboard & (1ULL << (square));
}

static inline void set_bit(uint64_t* bitboard, int square) {
    *bitboard |= (1ULL << square);
}

static inline void pop_bit(uint64_t* bitboard, int square) {
    *bitboard &= ~(1ULL << square);
}

// for my python
uint64_t get_bit_wrapper(uint64_t bitboard, int square);
void print_bitboard(uint64_t bitboard);
// De Brujin magic sequence stuff
extern const int bitscan_forward_magic_index[64];

static inline int bitscan_forward(uint64_t bb) {
#ifdef __GNUC__
    return __builtin_ctzll(bb);
#elif _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, bb);
    return index;
#else
    const uint64_t debruijn64 = 0x03f79d71b4cb0a89ULL;
    if (!bb)
        return -1;
    return bitscan_forward_magic_index[((bb & -bb) * debruijn64) >> 58];
#endif
}

static inline int popcount(uint64_t bb) {
#ifdef __GNUC__
    return __builtin_popcountll(bb);
#elif _MSC_VER
    return __popcnt64(bb);
#else
    bb = bb - ((bb >> 1) & 0x5555555555555555);
    bb = (bb & 0x3333333333333333) + ((bb >> 2) & 0x3333333333333333);
    bb = (bb + (bb >> 4)) & 0x0f0f0f0f0f0f0f0f;
    bb = (bb * 0x0101010101010101) >> 56;
    return (int)bb;
#endif
}

uint64_t index_to_uint64_t(int index, int num_bits, uint64_t attack_mask);

#endif
