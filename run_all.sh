#!/bin/bash

# 전체 파이프라인 실행 스크립트
# 사용법: ./run_all.sh [name] [num_runs]
# 예시: ./run_all.sh test 3
#      ./run_all.sh my_experiment 5

# 첫 번째 인자: name (기본값 "test")
# 두 번째 인자: num_runs (기본값 3)
NAME=${1:-test}
NUM_RUNS=${2:-3}

# 스크립트가 있는 디렉토리 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Training
bash "$SCRIPT_DIR/train.sh" "$NAME" "$NUM_RUNS"

# 2. Rendering
bash "$SCRIPT_DIR/render.sh" "$NAME" "$NUM_RUNS"

# 3. Testing mIoU
bash "$SCRIPT_DIR/test_miou.sh" "$NAME" "$NUM_RUNS"

