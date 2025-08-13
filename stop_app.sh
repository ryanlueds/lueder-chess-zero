#!/bin/bash
echo "Killing the python API server..."
pkill -f "python ./alphazero/src/api.py"

echo "Killing ALL vite processes..."
pkill -f "vite"

echo "Application stopped."
