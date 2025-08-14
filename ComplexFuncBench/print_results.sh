#!/bin/bash

# Function to find and display available models
find_available_models() {
    echo "Scanning for models available for report..."
    find result -maxdepth 4 -type d -name "full-1000" 2>/dev/null | sed 's|/full-1000||' | sed 's|^result/||' | sort
}

if [ -z "$1" ]; then
    echo "Available models:"
    find_available_models | while read model; do
        echo "  $model"
    done
    echo ""
    echo "Usage: $0 <MODEL_NAME>"
    echo "Example: $0 gpt-4o-2024-08-06"
    exit 1
fi

MODEL_NAME=$1

# Check if the model directory exists
if [ ! -d "result/$MODEL_NAME/full-1000" ]; then
    echo "Error: Model directory 'result/$MODEL_NAME/full-1000' not found!"
    echo ""
    echo "Available models:"
    find_available_models | while read model; do
        echo "  $model"
    done
    exit 1
fi

echo "Running print_results.py for model: $MODEL_NAME"
uv run print_results.py --log_dir result/$MODEL_NAME/full-1000/logs --result_dir result/$MODEL_NAME/full-1000.jsonl
