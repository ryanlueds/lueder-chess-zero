#include <stdio.h>
#include <string.h>
#include "../common/constants.h"
#include "../framework/chess/board/board.h"
#include "../framework/chess/movegeneration/masks_and_attacks.h"
#include "perft.h"
#include "test_macros.h"

typedef struct {
    const char* fen;
    // this is sorta low depth, but i compile my tests with debug compiler flags which is substantially slower
    // because no static inlining
    const long results[6];
} PerftPosition;

// Perft positions
const PerftPosition start_pos = {start_position, {0, 20, 400, 8902, 197281, 4865609}};
const PerftPosition kiwi_pos = {kiwi_position, {0, 48, 2039, 97862, 4085603, 193690690}};
const PerftPosition endgame_pos = {endgame_position, {0, 14, 191, 2812, 43238, 674624}};
const PerftPosition tricky_pos = {tricky_position, {0, 6, 264, 9467, 422333, 15833292}};
const PerftPosition talk_pos = {talk_position, {0, 44, 1486, 62379, 2103487, 89941194}};

int assert_position(const PerftPosition*, const char*);

int main(int argc, char* argv[]) {
    const PerftPosition* selected_position;
    const char* position_name;

    if (argc > 1) {
        if (strcmp(argv[1], "kiwi") == 0) {
            selected_position = &kiwi_pos;
            position_name = "kiwi";
        } else if (strcmp(argv[1], "endgame") == 0) {
            selected_position = &endgame_pos;
            position_name = "endgame";
        } else if (strcmp(argv[1], "tricky") == 0) {
            selected_position = &tricky_pos;
            position_name = "tricky";
        } else if (strcmp(argv[1], "talk") == 0) {
            selected_position = &talk_pos;
            position_name = "talk";
        } else {
            selected_position = &start_pos;
            position_name = "start";
        }

        printf("Running perft test suite for '%s' position...\n", position_name);
        assert_position(selected_position, position_name);

        if (total_failures > 0) {
            printf(COLOR_RED "  %d TEST(S) FAILED OUT OF %d\n" COLOR_RESET, total_failures, total_tests);
            printf(COLOR_RED "%s" COLOR_RESET, failed_tests);
        } else {
            printf(COLOR_GREEN "  ALL %d TESTS PASSED\n" COLOR_RESET, total_tests);
        }

        return 0;
    }

    selected_position = &kiwi_pos;
    position_name = "kiwi";
    printf("Running perft test suite for '%s' position...\n", position_name);
    assert_position(selected_position, position_name);

    selected_position = &endgame_pos;
    position_name = "endgame";
    printf("Running perft test suite for '%s' position...\n", position_name);
    assert_position(selected_position, position_name);

    selected_position = &tricky_pos;
    position_name = "tricky";
    printf("Running perft test suite for '%s' position...\n", position_name);
    assert_position(selected_position, position_name);

    selected_position = &talk_pos;
    position_name = "talk";
    printf("Running perft test suite for '%s' position...\n", position_name);
    assert_position(selected_position, position_name);

    selected_position = &start_pos;
    position_name = "start";
    printf("Running perft test suite for '%s' position...\n", position_name);
    assert_position(selected_position, position_name);

    if (total_failures > 0) {
        printf(COLOR_RED "  %d TEST(S) FAILED OUT OF %d\n" COLOR_RESET, total_failures, total_tests);
        printf(COLOR_RED "%s" COLOR_RESET, failed_tests);
    } else {
        printf(COLOR_GREEN "  ALL %d TESTS PASSED\n" COLOR_RESET, total_tests);
    }

    return 0;
}

int assert_position(const PerftPosition* selected_position, const char* position_name) {
    init_all_attack_tables();
    Board board;

    parse_fen(&board, (char*)selected_position->fen);

    for (int depth = 1; depth <= 5; depth++) {
        long nodes = perft_test(&board, depth);
        char msg[100];
        sprintf(msg, "Depth: %d, Position: %s", depth, position_name);
        EXPECT_EQ(nodes, selected_position->results[depth], msg);
    }

    return 0;
}