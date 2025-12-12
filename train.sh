#!/bin/bash

# 첫 번째 인자: name (기본값 "test")
# 두 번째 인자: num_runs (기본값 3)
NAME=${1:-test}
NUM_RUNS=${2:-3}

python train.py -s /ws/data/Ref-lerf/ramen -m /ws/data/Ref-lerf/train/ramen --name "$NAME" --num_runs "$NUM_RUNS"