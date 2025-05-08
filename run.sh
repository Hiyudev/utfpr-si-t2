#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No argument provided. Usage: ./run.sh <argument>"
    exit 1
fi

python3 main.py "$1"