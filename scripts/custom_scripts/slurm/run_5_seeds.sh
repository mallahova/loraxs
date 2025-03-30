#!/bin/bash

# Check if the script name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <script_name>"
  exit 1
fi

# Script name (first argument)
script_name="$1"

# Loop through seeds 0, 1, 2, 3, 4 and submit sbatch jobs
for seed in {0..4}; do
  sbatch "$script_name" "$seed"
done
