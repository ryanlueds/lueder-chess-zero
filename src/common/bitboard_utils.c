#include "bitboard_utils.h"
#include "stdio.h"

void print_bitboard(uint64_t bitboard) {
    printf("\n");
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square = 8 * rank + file;
            // printing ranks
            if (!file) {
                printf("  %d ", 8 - rank);
            }
            printf(" %d", get_bit(bitboard, square) ? 1 : 0);
        }
        printf("\n");
    }

    printf("\n     a b c d e f g h\n\n");
    printf("     Bitboard: %llud\n\n", bitboard);
}

uint64_t index_to_uint64_t(int index, int num_bits, uint64_t attack_mask) {
    uint64_t result = 0ULL;

    for (int i = 0; i < num_bits; i++) {
        int square = bitscan_forward(attack_mask);
        pop_bit(&attack_mask, square);
        if (index & (1 << i)) {
            set_bit(&result, square);
        }
    }

    return result;
}

// for my python
uint64_t get_bit_wrapper(uint64_t bitboard, int square) {
    return bitboard & (1ULL << (square));
}
