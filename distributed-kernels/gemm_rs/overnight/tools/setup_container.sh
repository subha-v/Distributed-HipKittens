#!/usr/bin/env bash
# Create a dedicated build/run container for the gfx942 GEMM-RS bring-up.
# Cloned device/ipc/security config from the existing campaign container, but
# runs as the invoking user so artifacts in $HOME stay user-owned.
set -euo pipefail

NAME=dhk-gemmrs
IMAGE=vllm/vllm-openai-rocm:v0.26.0

echo "host id: $(id -u):$(id -g)  groups: $(id -Gn)"
RENDER_GID=$(getent group render | cut -d: -f3 || true)
VIDEO_GID=$(getent group video | cut -d: -f3 || true)
echo "render gid=${RENDER_GID:-none} video gid=${VIDEO_GID:-none}"

if [ "$(docker ps -aq -f name=^${NAME}$)" != "" ]; then
  echo "removing existing ${NAME}"
  docker rm -f "${NAME}" >/dev/null
fi

docker run -d --name "${NAME}" \
  --entrypoint /bin/bash \
  --device=/dev/kfd --device=/dev/dri \
  ${VIDEO_GID:+--group-add ${VIDEO_GID}} \
  ${RENDER_GID:+--group-add ${RENDER_GID}} \
  --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --user "$(id -u):$(id -g)" \
  -v /home/subvadla:/home/subvadla \
  -e HOME=/home/subvadla \
  -w /home/subvadla/dhk \
  "${IMAGE}" -c 'sleep infinity' >/dev/null

sleep 3
docker ps -f name=^${NAME}$ --format 'status: {{.Status}}'
echo "== container up =="
docker exec "${NAME}" bash -c 'id; python3 -V; python3 -c "import torch;print(torch.__version__, torch.cuda.device_count())"; hipcc --version | head -1; python3 -c "import pybind11;print(pybind11.get_include())"; python3-config --includes'
