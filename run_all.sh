#!/usr/bin/env bash
set -uo pipefail  # Remove 'set -e' to prevent premature exit on errors

ratios=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25
        2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75
        5 5.25 5.5 5.75 6 6.25 6.5 6.75 7 7.25
        7.5 7.75 8 10 15 20 50 75 100)
exe=./main
dataset="Hypergraphs/com-amazon-cmty-hygra"
logfile="Results/result-com-amazon-cmty-hygra.log"

# Ensure the Results directory exists
mkdir -p "$(dirname "$logfile")"

# Clear the logfile
: > "$logfile"

for pct in "${ratios[@]}"; do
  echo "=== Running $exe $pct $dataset 0 0 0 ===" | tee -a "$logfile"
  
  # Run the command and capture output, continue on error
  if ! "$exe" "$pct" "$dataset" 0 0 0 2>&1 | tee -a "$logfile"; then
    echo "ERROR: $exe failed for ratio $pct with exit code $?" | tee -a "$logfile"
  fi
done

echo "Completed processing all ratios." | tee -a "$logfile"