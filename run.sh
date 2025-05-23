#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No argument provided. Usage: ./run.sh <argument>"
    exit 1
fi

# Check if the first argument is not "plot"
if [ "$1" == "plot" ]; then
    python3 plot.py
    exit 0
fi

python3 main.py "$1" "$2"