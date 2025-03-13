#!/bin/bash

# Enable strict error handling
set -e  # Exit script if any command fails
set -o pipefail  # Catch errors in piped commands

# Define datasets
datasets=("DSA" "AI_Act" "GDPR")

# Function to run the script with a given dataset
run_script() {
    dataset=$1
    echo "==================================================="
    echo "Starting main.py with dataset: $dataset"
    echo "==================================================="

    # Run the script and log output
    python main.py --dataset="$dataset"

    echo "✅ Completed: $dataset"
    echo "---------------------------------------------------"
}

# Loop through datasets and run main.py for each
for dataset in "${datasets[@]}"; do
    run_script "$dataset"
done

echo "All datasets processed successfully!"
