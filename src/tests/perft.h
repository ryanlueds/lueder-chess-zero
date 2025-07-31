#ifndef PERFT_H
#define PERFT_H

#include <stdio.h>
#include <sys/time.h>

#include "../framework/chess/board/board.h"
#include "../framework/chess/movegeneration/move_generation.h"

extern long nodes;

long perft_test(Board* board, int depth);
void perft_with_output(Board* board, int depth);

static inline void perft_driver(Board* board, int depth) {
    Moves move_list;
    generate_moves(board, &move_list);
    MoveData data;

    if (depth == 1) {
        nodes += move_list.count;
        return;
    }

    for (int i = 0; i < move_list.count; i++) {
        data.move = move_list.moves[i];
        data.castling_rights = board->castle;
        data.enpassant_square = board->enpassant;

        data.occupancies[0] = board->occupancies[0];
        data.occupancies[1] = board->occupancies[1];
        data.occupancies[2] = board->occupancies[2];

        data.captured_piece = make_move(board, data.move);

        if (depth == 2) {
            Moves move_list_d2;
            generate_moves(board, &move_list_d2);
            nodes += move_list_d2.count;
        } else {
            perft_driver(board, depth - 1);
        }

        undo_move(board, &data);
    }
}

int get_time_ms();

#endif
