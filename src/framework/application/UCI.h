#ifndef UCI_H
#define UCI_H

#include "../../common/bitboard_utils.h"
#include "../../common/constants.h"
#include "../../common/move_encoding.h"
#include "../chess/board/board.h"
#include "../chess/movegeneration/move_generation.h"
#include "../chess/search/search.h"

int parse_move(Board* board, char* move_string);
void parse_position(Board* board, char* command);
void parse_go(Board* board, char* command);
void uci_loop();

#endif
