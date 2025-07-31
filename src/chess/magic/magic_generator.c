#include <stdio.h>
#include <stdlib.h>

#include "../../chess/attacks/static_attacks.h"
#include "../../common/bitboard_utils.h"
#include "../../common/types.h"
#include "magic_generator.h"

#define USE_32_BIT_MULTIPLICATIONS

unsigned int random_state = 91827319;

unsigned int get_U32_random_number() {
    unsigned int number = random_state;

    // XOR shift algorithm
    number ^= number << 13;
    number ^= number >> 17;
    number ^= number << 5;

    random_state = number;

    return number;
}

uint64_t get_uint64_t_random_number() {
    uint64_t n1, n2, n3, n4;
    n1 = (uint64_t)(get_U32_random_number() & 0xFFFF);
    n2 = (uint64_t)(get_U32_random_number() & 0xFFFF);
    n3 = (uint64_t)(get_U32_random_number() & 0xFFFF);
    n4 = (uint64_t)(get_U32_random_number() & 0xFFFF);

    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48);
}

uint64_t random_uint64_fewbits() {
    return get_uint64_t_random_number() & get_uint64_t_random_number() & get_uint64_t_random_number();
}

int transform(uint64_t b, uint64_t magic, int bits) {
#if defined(USE_32_BIT_MULTIPLICATIONS)
    return (unsigned)((int)b * (int)magic ^ (int)(b >> 32) * (int)(magic >> 32)) >> (32 - bits);
#else
    return (int)((b * magic) >> (64 - bits));
#endif
}

uint64_t find_magic(int sq, int m, int bishop) {
    uint64_t mask, b[4096], a[4096], used[4096], magic;
    int i, j, k, n, fail;

    mask = bishop ? get_bishop_mask(sq) : get_rook_mask(sq);
    n = popcount(mask);

    for (i = 0; i < (1 << n); i++) {
        b[i] = index_to_uint64_t(i, n, mask);
        a[i] = bishop ? get_bishop_attack(sq, b[i]) : get_rook_attack(sq, b[i]);
    }
    for (k = 0; k < 100000000; k++) {
        magic = random_uint64_fewbits();
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6)
            continue;
        for (i = 0; i < 4096; i++)
            used[i] = 0ULL;
        for (i = 0, fail = 0; !fail && i < (1 << n); i++) {
            j = transform(b[i], magic, m);
            if (used[j] == 0ULL)
                used[j] = a[i];
            else if (used[j] != a[i])
                fail = 1;
        }
        if (!fail)
            return magic;
    }
    printf("***Failed***\n");
    return 0ULL;
}
