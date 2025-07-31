#include "./framework/chess/board/board.h"
#include "./framework/chess/movegeneration/masks_and_attacks.h"
#include "./tests/perft.h"

int main() {
    Board board[1];
    init_all_attack_tables();
    parse_fen(board, start_position);
    perft_with_output(board, 6);

    return 0;
}
