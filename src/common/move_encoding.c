#include "move_encoding.h"

// for my python code---it doesnt like static inlines
int get_move_source_wrapper(int move)     { return move & 0b111111; }
int get_move_target_wrapper(int move)     { return (move >> 6) & 0b111111; }
int get_move_piece_wrapper(int move)      { return (move >> 12) & 0b1111; }
int get_move_promoted_wrapper(int move)   { return (move >> 16) & 0b1111; }
int get_move_capture_wrapper(int move)    { return move & 0b000100000000000000000000; }
int get_move_double_wrapper(int move)     { return move & 0b001000000000000000000000; }
int get_move_enpassant_wrapper(int move)  { return move & 0b010000000000000000000000; }
int get_move_castling_wrapper(int move)   { return move & 0b100000000000000000000000; }
