#!/bin/bash

# ReferSplat Docker 실행 스크립트
# 사용법: ./run_docker.sh [GPU번호] [명령어]
# 예시: ./run_docker.sh 0 python train.py ...
#      ./run_docker.sh 1 bash
#      ./run_docker.sh python train.py ...   (기본값: GPU 0)

set -e

# 기본 설정
IMAGE_NAME="wanheekim/refersplat:latest"
# CONTAINER_NAME은 이제 GPU 번호를 포함하여 고유하게 설정됨
WORK_DIR="/ws/external"

# 현재 스크립트가 있는 디렉토리의 상위 디렉토리 (ReferSplat 프로젝트 루트)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 데이터셋 경로
DATASET_DIR="/media/TrainDataset/wh"
DATASET_MOUNT="/ws/data"

# GPU 번호 파싱 (첫 번째 인자가 숫자 또는 쉼표로 구분된 숫자인지 확인)
GPU_NUM="0"
COMMAND_ARGS=()

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    # 첫 번째 인자가 숫자 또는 쉼표로 구분된 숫자면 GPU 번호로 사용
    GPU_NUM="$1"
    shift
    COMMAND_ARGS=("$@")
else
    # 첫 번째 인자가 숫자가 아니면 모든 인자를 명령어로 사용 (기본 GPU 0)
    COMMAND_ARGS=("$@")
fi

# =========================================================
# ⭐ 수정된 부분: CONTAINER_NAME을 GPU 번호를 기반으로 설정
# 예: GPU_NUM=0 -> CONTAINER_NAME=refersplat_0
# 예: GPU_NUM=0,1 -> CONTAINER_NAME=refersplat_0_1
CONTAINER_NAME="refersplat_${GPU_NUM//,/_}"
# =========================================================


# GPU 사용 여부 확인 및 설정
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus device=${GPU_NUM}"
    echo "GPU detected. Using GPU(s): ${GPU_NUM}"
else
    GPU_FLAG=""
    echo "No GPU detected. Running in CPU mode"
fi

# =========================================================
# ⭐ 수정된 부분: 이제 현재 스크립트가 실행하려는 고유한 컨테이너만 중지/제거 시도
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping existing container named: ${CONTAINER_NAME}..."
    docker stop ${CONTAINER_NAME} > /dev/null 2>&1 || true
    docker rm ${CONTAINER_NAME} > /dev/null 2>&1 || true
fi
# =========================================================

# 도커 이미지 pull (최신 버전 확인)
echo "Pulling Docker image: ${IMAGE_NAME}"
docker pull ${IMAGE_NAME}

# 도커 컨테이너 실행
echo "Starting Docker container: ${CONTAINER_NAME}"
echo "Project directory: ${PROJECT_DIR} -> ${WORK_DIR}"
echo "Dataset directory: ${DATASET_DIR} -> ${DATASET_MOUNT}"
echo "GPU(s): ${GPU_NUM}"

# 명령어가 제공되지 않으면 bash로 실행
if [ ${#COMMAND_ARGS[@]} -eq 0 ]; then
    echo "No command provided. Starting interactive bash shell..."
    docker run -it --rm \
        ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        --ipc=host \
        --privileged \
        -v "${PROJECT_DIR}":${WORK_DIR} \
        -v "${DATASET_DIR}":${DATASET_MOUNT} \
        -w ${WORK_DIR} \
        -e DISPLAY=${DISPLAY:-} \
        -e PYTHONUNBUFFERED=1 \
        -e CUDA_VISIBLE_DEVICES=${GPU_NUM} \
        ${IMAGE_NAME} \
        bash
else
    # 제공된 명령어 실행
    docker run -it --rm \
        ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        --ipc=host \
        --privileged \
        -v "${PROJECT_DIR}":${WORK_DIR} \
        -v "${DATASET_DIR}":${DATASET_MOUNT} \
        -w ${WORK_DIR} \
        -e DISPLAY=${DISPLAY:-} \
        -e PYTHONUNBUFFERED=1 \
        -e CUDA_VISIBLE_DEVICES=${GPU_NUM} \
        ${IMAGE_NAME} \
        "${COMMAND_ARGS[@]}"
fi

echo "Container stopped."