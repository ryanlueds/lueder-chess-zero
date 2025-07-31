#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "types.h"

// FEN debug positions
#define empty_board "8/8/8/8/8/8/8/8 w - - "
#define start_position "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
#define kiwi_position "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
#define endgame_position "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 "
#define tricky_position "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
#define talk_position "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

extern const char* square_to_coordinates[65];
extern char* unicode_pieces[12];
extern int char_pieces[128];
extern char promoted_pieces[12];

#endif
