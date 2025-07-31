cmake_minimum_required(VERSION 4.0.3)

file(GLOB_RECURSE CORE_SRCS CONFIGURE_DEPENDS "*.c")

set(MAIN_SRC main.c)
set(TEST_SRC tests/test_runner.c)
set(PERFT_SRC tests/perft_runner.c)

list(REMOVE_ITEM CORE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/${PERFT_SRC}
)

include_directories(
    ./
    ./common
    ./chess/attacks
    ./framework/application
    ./framework/chess/board
    ./framework/chess/movegeneration
    ./framework/chess/search
    ./tests
)

set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -flto -fno-exceptions -fomit-frame-pointer -DNDEBUG")

add_library(chess_engine SHARED ${CORE_SRCS})

add_executable(lueder-computer ${MAIN_SRC})
target_link_libraries(lueder-computer chess_engine)

add_executable(test_runner ${TEST_SRC})
target_link_libraries(test_runner chess_engine)

add_executable(perft_runner ${PERFT_SRC})
target_link_libraries(perft_runner chess_engine)

add_custom_target(test
    COMMAND test_runner
    DEPENDS test_runner
)

add_custom_target(perft
    COMMAND perft_runner
    DEPENDS perft_runner
)
