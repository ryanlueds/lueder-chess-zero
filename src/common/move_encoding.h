#ifndef MOVE_ENCODING_H
#define MOVE_ENCODING_H

/*
          binary move bits                               hexidecimal constants

    0000 0000 0000 0000 0011 1111    source square       0x3f
    0000 0000 0000 1111 1100 0000    target square       0xfc0
    0000 0000 1111 0000 0000 0000    piece               0xf000
    0000 1111 0000 0000 0000 0000    promoted piece      0xf0000
    0001 0000 0000 0000 0000 0000    capture flag        0x100000
    0010 0000 0000 0000 0000 0000    double push flag    0x200000
    0100 0000 0000 0000 0000 0000    enpassant flag      0x400000
    1000 0000 0000 0000 0000 0000    castling flag       0x800000
*/

// clang-format off
static inline int encode_move(int source, int target, int piece, int promoted, int capture, int double_push, int enpassant, int castling) {
    return source               |
           (target << 6)        |
           (piece << 12)        |
           (promoted << 16)     |
           (capture << 20)      |
           (double_push << 21)  |
           (enpassant << 22)    |
           (castling << 23);
}

// extract source square
static inline int get_move_source(int move)     { return move & 0b111111; }
// extract target square
static inline int get_move_target(int move)     { return (move >> 6) & 0b111111; }
// extract piece
static inline int get_move_piece(int move)      { return (move >> 12) & 0b1111; }
// extract promoted piece
static inline int get_move_promoted(int move)   { return (move >> 16) & 0b1111; }
// extract capture flag
static inline int get_move_capture(int move)    { return move & 0b000100000000000000000000; }
// extract double pawn push flag
static inline int get_move_double(int move)     { return move & 0b001000000000000000000000; }
// extract enpassant flag
static inline int get_move_enpassant(int move)  { return move & 0b010000000000000000000000; }
// extract castling flag
static inline int get_move_castling(int move)   { return move & 0b100000000000000000000000; }
// clang-format on

// for my python code
int get_move_source_wrapper(int move);
int get_move_target_wrapper(int move);
int get_move_piece_wrapper(int move);
int get_move_promoted_wrapper(int move);
int get_move_capture_wrapper(int move);
int get_move_double_wrapper(int move);
int get_move_enpassant_wrapper(int move);
int get_move_castling_wrapper(int move);

#endif
