# 1. 베이스 이미지 변경 (Ubuntu 20.04 기반의 NVIDIA 공식 이미지 사용)
# 기존: FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel (Ubuntu 18.04 문제)
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 설치 (Python 3.8 및 필수 도구 추가)
# Ubuntu 20.04는 기본 Python이 3.8입니다.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version | grep -E "Python 3\.[89]|Python 3\.1[0-9]" || (echo "ERROR: Python 3.8+ required but found: $(python3 --version)" && exit 1)

# Python 버전 확인 (3.8 이상이어야 함)
RUN python3 --version

# 시스템 Python이 conda보다 우선하도록 PATH 설정
# python3를 python으로 사용하기 위한 심볼릭 링크 (편의상)
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python --version

# PATH에서 시스템 Python이 conda보다 우선하도록 설정
ENV PATH="/usr/bin:${PATH}"

# 4. Pip 업그레이드
RUN pip3 install --upgrade pip

# 5. Python 3.8과 호환되는 typing-extensions 버전 먼저 설치
# (PyTorch 설치 시 최신 버전이 설치되어 Python 3.9+ 요구로 인한 에러 방지)
RUN pip install "typing-extensions<5.0"

# 6. PyTorch 1.12.1 수동 설치 (기존 베이스 이미지에 있던 것 대체)
# CUDA 11.3 버전에 맞는 PyTorch 설치
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# ---------------- 아래는 기존과 동일 ----------------

# 7. 의존성 패키지 설치 (jaxtyping, typeguard 등)
RUN pip install typeguard==4.0.0 jaxtyping==0.2.12

# 8. Rust 및 기타 패키지 설치
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    pip install maturin==0.14.13 && \
    pip install --no-build-isolation \
        open-clip-torch \
        plyfile==0.8.1 \
        tqdm \
        transformers \
        opencv-python \
        tensorboard \
        matplotlib

# 9. 로컬 submodules 복사
COPY submodules /app/submodules

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

# 10. 서브모듈 설치
RUN pip install /app/submodules/segment-anything-langsplat
RUN pip install /app/submodules/langsplat-rasterization
RUN pip install /app/submodules/simple-knn