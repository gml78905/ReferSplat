#!/bin/bash
set -e

# 1. 기본 설정
USER_NAME=$(whoami)
# 요청하신 이미지 이름 적용
IMAGE_NAME="wanheekim/refersplat:latest"
CONTAINER_NAME="${USER_NAME}_refersplat_container"

# 2. 경로 설정
# HOST_WS_DIR: 현재 디렉토리 (realpath 사용)
HOST_WS_DIR=$(realpath .)

# DATA_DIR: 요청하신 데이터셋 경로
DATA_DIR="/media/gml78905/T71/dataset" 

# 컨테이너 내부 마운트 경로 (요청하신 경로로 변경)
WORKSPACE_DIR="/ws/external"
DATASET_DIR="/ws/data"

echo "================================================"
echo "USER: $USER_NAME"
echo "IMAGE: $IMAGE_NAME"
echo "CONTAINER: $CONTAINER_NAME"
echo "HOST WORKSPACE: $HOST_WS_DIR -> $WORKSPACE_DIR"
echo "DATA DIR: $DATA_DIR -> $DATASET_DIR"
echo "================================================"

# 3. Docker 이미지 빌드 (현재 Dockerfile 기반으로 해당 태그명으로 빌드)
# 만약 빌드 없이 기존 이미지를 바로 쓰고 싶다면 이 줄을 주석 처리하세요.
echo "🚀 Building Docker Image..."
DOCKER_BUILDKIT=1 docker build -t $IMAGE_NAME .

# 4. 기존 컨테이너 정리 (혹시 실행 중이거나 남아있다면 삭제)
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🧹 Removing existing container..."
    docker rm -f $CONTAINER_NAME
fi

# 5. 컨테이너 실행
echo "🚀 Starting container '$CONTAINER_NAME'..."

# 요청하신 플래그와 환경 변수 적용
docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 64G \
  --cpus=$(nproc) \
  --ipc=host \
  --pid=host \
  \
  -e DISPLAY="unix$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOST_WS_DIR":"$WORKSPACE_DIR" \
  -v "$DATA_DIR":"$DATASET_DIR" \
  \
  -w "$WORKSPACE_DIR" \
  "$IMAGE_NAME" \
  bash
