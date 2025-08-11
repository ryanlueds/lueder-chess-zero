from wrapper import (
    get_move_source, get_move_target, get_move_promoted
)


class bcolors:
    HEADER = '[95m'
    OKBLUE = '[94m'
    OKCYAN = '[96m'
    OKGREEN = '[92m'
    WARNING = '[93m'
    FAIL = '[91m'
    ENDC = '[0m'
    BOLD = '[1m'
    UNDERLINE = '[4m'

def square_to_str(sq):
    rank = sq // 8
    file = sq % 8
    real_rank = 7 - rank
    real_sq = real_rank * 8 + file
    f = "abcdefgh"[real_sq % 8]
    r = "12345678"[real_sq // 8]
    return f + r

def move_to_str(move):
    source_sq = get_move_source(move)
    target_sq = get_move_target(move)
    promoted_piece = get_move_promoted(move)

    move_str = square_to_str(source_sq) + square_to_str(target_sq)

    if promoted_piece:
        promoted_char = {1: 'n', 2: 'b', 3: 'r', 4: 'q',
                         7: 'n', 8: 'b', 9: 'r', 10: 'q'}.get(promoted_piece, '')
        move_str += promoted_char
    return move_str



def convert_channel_to_bitboard(tensor_channel):
    bitboard = 0
    for r in range(8):
        for f in range(8):
            if tensor_channel[r][f] == 1:
                bitboard |= (1 << (r * 8 + f))
    return bitboard
