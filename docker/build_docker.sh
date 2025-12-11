#!/bin/bash

# ReferSplat Docker 이미지 빌드 스크립트
# 사용법: ./build_docker.sh [이미지태그]
# 예시: ./build_docker.sh
#      ./build_docker.sh refersplat:local
#      ./build_docker.sh wanheekim/refersplat:v1.0

set -e

# 현재 스크립트가 있는 디렉토리의 상위 디렉토리 (ReferSplat 프로젝트 루트)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 이미지 태그 설정 (기본값: refersplat:latest)
IMAGE_TAG="${1:-refersplat:latest}"

echo "Building Docker image: ${IMAGE_TAG}"
echo "Project directory: ${PROJECT_DIR}"
echo "Dockerfile location: ${PROJECT_DIR}/Dockerfile"

# Docker 이미지 빌드
docker build -t "${IMAGE_TAG}" "${PROJECT_DIR}"

echo ""
echo "Docker image built successfully: ${IMAGE_TAG}"
echo ""
echo "To run the container, use:"
echo "  bash docker/run_docker.sh --image ${IMAGE_TAG} [GPU번호] [명령어]"
echo ""
echo "Or modify docker/run_docker.sh to use this image by default."

