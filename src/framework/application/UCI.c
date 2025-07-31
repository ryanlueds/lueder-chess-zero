#include "UCI.h"
#include <string.h>

int parse_move(Board* board, char* move_string) {
    Moves move_list[1];
    generate_moves(board, move_list);
    int source_square = (move_string[0] - 'a') + (8 - (move_string[1] - '0')) * 8;

    int target_square = (move_string[2] - 'a') + (8 - (move_string[3] - '0')) * 8;

    for (int move_count = 0; move_count < move_list->count; move_count++) {
        int move = move_list->moves[move_count];

        // make sure source & target squares are available within the generated move
        if (source_square == get_move_source(move) && target_square == get_move_target(move)) {
            int promoted_piece = get_move_promoted(move);

            if (promoted_piece) {
                if ((promoted_piece == Q || promoted_piece == q) && move_string[4] == 'q') {
                    return move;
                } else if ((promoted_piece == R || promoted_piece == r) && move_string[4] == 'r') {
                    return move;
                } else if ((promoted_piece == B || promoted_piece == b) && move_string[4] == 'b') {
                    return move;
                } else if ((promoted_piece == N || promoted_piece == n) && move_string[4] == 'n') {
                    return move;
                }
                // continue the loop on possible wrong promotions like e7e8f
                continue;
            }

            return move;
        }
    }

    // return illegal move
    return 0;
}

void parse_position(Board* board, char* command) {
    // shift pointer to the right where next token begins
    command += 9;

    char* current_char = command;

    if (strncmp(command, "startpos", 8) == 0) {
        parse_fen(board, start_position);
    } else {
        current_char = strstr(command, "fen");
        if (current_char == NULL) {
            parse_fen(board, start_position);
        } else {
            current_char += 4;
            parse_fen(board, current_char);
        }
    }

    current_char = strstr(command, "moves");

    if (current_char != NULL) {
        current_char += 6;

        while (*current_char) {
            int move = parse_move(board, current_char);

            if (move == 0)
                break;

            make_move(board, move);

            while (*current_char && *current_char != ' ')
                current_char++;

            current_char++;
        }
    }

    print_board(board);
}

void parse_go(Board* board, char* command) {
    int depth = -1;
    char* current_depth = NULL;

    if (current_depth == strstr(command, "depth")) {
        depth = atoi(current_depth + 6);
    } else {
        depth = 6;
    }

    search_position(board, depth);
}

void uci_loop() {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    // define user / GUI input buffer
    char input[2000];

    Board board[1];
    init_board(board);

    printf("id name LuederComputer\n");
    printf("id name Ryan Lueder\n");
    printf("uciok\n");

    while (1) {
        memset(input, 0, sizeof(input));

        fflush(stdout);

        if (!fgets(input, 2000, stdin))
            continue;

        if (input[0] == '\n')
            continue;

        if (strncmp(input, "isready", 7) == 0) {
            printf("readyok\n");
            continue;
        } else if (strncmp(input, "position", 8) == 0) {
            parse_position(board, input);
        } else if (strncmp(input, "ucinewgame", 10) == 0) {
            parse_position(board, "position startpos");
        } else if (strncmp(input, "go", 2) == 0) {
            parse_go(board, input);
        } else if (strncmp(input, "quit", 4) == 0) {
            break;
        } else if (strncmp(input, "uci", 3) == 0) {
            // TODO: idk
        }
    }
}
