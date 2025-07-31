#include <stdio.h>
#include <stdlib.h>
#include "../framework/chess/movegeneration/masks_and_attacks.h"
#include "test_macros.h"
#include "test_suite.h"

void run_tests() {
    printf("Running tests...\n");
    init_all_attack_tables();
    test_is_in_check();
    test_get_pinned_pieces();
    test_move_generation();

    printf("\nTest summary:\n");
    if (total_failures == 0) {
        printf(COLOR_GREEN "  ALL %d TESTS PASSED\n" COLOR_RESET, total_tests);
    } else {
        printf(COLOR_RED "  %d TEST(S) FAILED OUT OF %d\n" COLOR_RESET, total_failures, total_tests);
        printf(COLOR_RED "%s" COLOR_RESET, failed_tests);
        exit(1);
    }
}

int main() {
    run_tests();
    return 0;
}
