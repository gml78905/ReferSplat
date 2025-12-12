#!/bin/bash
MODEL_PATH="/ws/data/Ref-lerf/train/ramen"

# 첫 번째 인자: name (기본값 "test")
# 두 번째 인자: num_runs (기본값 3)
NAME=${1:-test}
NUM_RUNS=${2:-3}

for run_number in $(seq 1 $NUM_RUNS); do
    echo "Processing run $run_number/$NUM_RUNS"
    for iteration in {0..4}; do
        python render.py -m "$MODEL_PATH" --name "$NAME" --run_number "$run_number" --iteration "$iteration"
    done
done