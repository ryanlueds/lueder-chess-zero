#include "board.h"
#include "../../../common/bitboard_utils.h"
#include "../../../common/constants.h"

void init_board(Board* board) {
    parse_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void print_board_debug(const Board* board) {
    print_board(board);

    for (int piece = P; piece < k; piece++) {
        printf("BITBOARD FOR %s\n", unicode_pieces[piece]);
        print_bitboard(board->bitboards[piece]);
    }
}

void print_board(const Board* board) {
    printf("\n");

    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;

            if (!file)
                printf("  %d ", 8 - rank);

            int piece = -1;

            for (int bb_piece = P; bb_piece <= k; bb_piece++) {
                if (get_bit(board->bitboards[bb_piece], square))
                    piece = bb_piece;
            }

            printf(" %s", (piece == -1) ? "." : unicode_pieces[piece]);
        }
        printf("\n");
    }

    printf("\n     a b c d e f g h\n\n");
    printf("     Side:     %s\n", !board->side ? "white" : "black");
    printf("     Enpassant:   %s\n", (board->enpassant != no_sq) ? square_to_coordinates[board->enpassant] : "no");
    printf("     Castling:  %c%c%c%c\n\n", (board->castle & wk) ? 'K' : '-', (board->castle & wq) ? 'Q' : '-',
           (board->castle & bk) ? 'k' : '-', (board->castle & bq) ? 'q' : '-');
}

void parse_fen(Board* board, char* fen) {
    memset(board->bitboards, 0ULL, sizeof(board->bitboards));
    memset(board->occupancies, 0ULL, sizeof(board->occupancies));

    board->side = 0;
    board->enpassant = no_sq;
    board->castle = 0;

    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;

            if ((*fen >= 'a' && *fen <= 'z') || (*fen >= 'A' && *fen <= 'Z')) {
                int piece = char_pieces[*fen];
                set_bit(&board->bitboards[piece], square);
                fen++;
            }

            if (*fen >= '0' && *fen <= '9') {
                int offset = *fen - '0';
                int piece = -1;

                for (int bb_piece = P; bb_piece <= k; bb_piece++) {
                    if (get_bit(board->bitboards[bb_piece], square))
                        piece = bb_piece;
                }

                if (piece == -1)
                    file--;

                file += offset;
                fen++;
            }

            if (*fen == '/')
                fen++;
        }
    }

    fen++;
    (*fen == 'w') ? (board->side = white) : (board->side = black);
    fen += 2;

    while (*fen != ' ') {
        switch (*fen) {
            case 'K':
                board->castle |= wk;
                break;
            case 'Q':
                board->castle |= wq;
                break;
            case 'k':
                board->castle |= bk;
                break;
            case 'q':
                board->castle |= bq;
                break;
            case '-':
                break;
        }

        fen++;
    }

    fen++;

    if (*fen != '-') {
        int file = fen[0] - 'a';
        int rank = 8 - (fen[1] - '0');
        board->enpassant = rank * 8 + file;
    }

    else
        board->enpassant = no_sq;

    for (int piece = P; piece <= K; piece++) {
        board->occupancies[white] |= board->bitboards[piece];
    }

    for (int piece = p; piece <= k; piece++) {
        board->occupancies[black] |= board->bitboards[piece];
    }

    board->occupancies[both] |= board->occupancies[white];
    board->occupancies[both] |= board->occupancies[black];
}
