#ifndef MAGIC_GENERATOR_H
#define MAGIC_GENERATOR_H

#include "../../common/types.h"

// magic square stuff. this is for precomputing masks and attack tables
uint64_t find_magic(int sq, int m, int bishop);

#endif
