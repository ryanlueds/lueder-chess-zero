#!/bin/bash
python ./alphazero/src/api.py &
cd ./chess-frontend && npm run dev
