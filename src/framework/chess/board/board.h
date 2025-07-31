#ifndef BOARD_H
#define BOARD_H

#include <string.h>
#include "../../../common/constants.h"
#include "../../../common/types.h"

typedef struct {
    uint64_t bitboards[12];
    uint64_t occupancies[3];
    int side;
    int enpassant;
    int castle;
} Board;

void print_board(const Board* board);
void print_board_debug(const Board* board);
void parse_fen(Board* board, char* fen);
void init_board(Board* board);

#endif
