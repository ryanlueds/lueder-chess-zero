#include "perft.h"

long nodes;

int get_time_ms() {
    struct timeval time_value;
    gettimeofday(&time_value, NULL);
    return time_value.tv_sec * 1000 + time_value.tv_usec / 1000;
}

long perft_test(Board* board, int depth) {
    nodes = 0;
    perft_driver(board, depth);
    return nodes;
}

void perft_with_output(Board* board, int depth) {
    printf("\nPerformance test:\n");

    long start = get_time_ms();
    Moves move_list[1];
    generate_moves(board, move_list);

    for (int move_count = 0; move_count < move_list->count; move_count++) {
        Board copy_board_state;
        copy_board(board, &copy_board_state);

        make_move(board, move_list->moves[move_count]);

        long cummulative_nodes = nodes;

        perft_driver(board, depth - 1);

        long old_nodes = nodes - cummulative_nodes;

        take_back(board, &copy_board_state);

        printf("move: %s%s%c  nodes: %ld\n", square_to_coordinates[get_move_source(move_list->moves[move_count])],
               square_to_coordinates[get_move_target(move_list->moves[move_count])],
               get_move_promoted(move_list->moves[move_count])
                   ? promoted_pieces[get_move_promoted(move_list->moves[move_count])]
                   : ' ',
               old_nodes);
    }

    printf("\nDepth: %d\n", depth);
    printf("Nodes: %ld\n", nodes);
    printf("Time: %ldms\n\n", get_time_ms() - start);
}
