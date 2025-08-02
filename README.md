# LuederComputer

chess engine

## Features
- [x] bitboards with magic square stuff
- [x] pretty fast move gen
- [ ] mcts that actually works
- [ ] working python code

## Build
```
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
```

```
# perft(7) (make sure you use release compiler flags)
./build/bin/lueder-computer

# tests all https://www.chessprogramming.org/Perft_Results
./build/bin/perft_runner

# tests for move generation edge cases (tons of edge cases...)
./build/bin/test_runner
```


## perft(7)
```
Performance test:
move: a2a3   nodes: 106743106
move: a2a4   nodes: 137077337
move: b2b3   nodes: 133233975
move: b2b4   nodes: 134087476
move: c2c3   nodes: 144074944
move: c2c4   nodes: 157756443
move: d2d3   nodes: 227598692
move: d2d4   nodes: 269605599
move: e2e3   nodes: 306138410
move: e2e4   nodes: 309478263
move: f2f3   nodes: 102021008
move: f2f4   nodes: 119614841
move: g2g3   nodes: 135987651
move: g2g4   nodes: 130293018
move: h2h3   nodes: 106678423
move: h2h4   nodes: 138495290
move: b1a3   nodes: 120142144
move: b1c3   nodes: 148527161
move: g1f3   nodes: 147678554
move: g1h3   nodes: 120669525

Depth: 7
Nodes: 3195901860
Time: 6828ms
```
