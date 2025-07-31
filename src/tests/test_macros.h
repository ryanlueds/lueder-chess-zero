#ifndef TEST_MACROS_H
#define TEST_MACROS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"

// Global failure tracking
extern int total_failures;
extern int total_tests;
extern char failed_tests[1024];

#define FAIL(msg, expected, actual)                                                                                    \
    do {                                                                                                               \
        total_failures++;                                                                                              \
        char buffer[256];                                                                                              \
        sprintf(buffer, "    " COLOR_RED "✗ %s: EXPECTED %lld, GOT %lld" COLOR_RESET "\n", msg, (long long)(expected), \
                (long long)(actual));                                                                                  \
        strcat(failed_tests, buffer);                                                                                  \
        printf("%s", buffer);                                                                                          \
    } while (0)

#define FAIL_BOOL(msg, expected)                                                            \
    do {                                                                                    \
        total_failures++;                                                                   \
        char buffer[256];                                                                   \
        sprintf(buffer, "    " COLOR_RED "✗ %s: EXPECTED %s, GOT %s" COLOR_RESET "\n", msg, \
                expected ? "true" : "false", expected ? "false" : "true");                  \
        strcat(failed_tests, buffer);                                                       \
        printf("%s", buffer);                                                               \
    } while (0)

#define PASS(msg)                                                        \
    do {                                                                 \
        printf("    " COLOR_GREEN "✓ %s: PASSED" COLOR_RESET "\n", msg); \
    } while (0)

#define EXPECT_EQ(actual, expected, msg) \
    do {                                 \
        total_tests++;                   \
        if ((actual) != (expected)) {    \
            FAIL(msg, expected, actual); \
        } else {                         \
            PASS(msg);                   \
        }                                \
    } while (0)

#define EXPECT_TRUE(cond, msg)    \
    do {                          \
        total_tests++;            \
        if (!(cond)) {            \
            FAIL_BOOL(msg, true); \
        } else {                  \
            PASS(msg);            \
        }                         \
    } while (0)

#define EXPECT_NONZERO(val, msg) \
    do {                         \
        total_tests++;           \
        if ((val) == 0) {        \
            FAIL(msg, 1, 0);     \
        } else {                 \
            PASS(msg);           \
        }                        \
    } while (0)

#endif
