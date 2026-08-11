#!/usr/bin/env bash
# Benchmark the frozen rank-1 MI300X submission with the OFFICIAL evaluator.
#
# Compatibility repairs needed on this node, and why each is the least invasive
# option. The frozen submission is NOT edited.
#
#   1. rank-1 does `import iris` and calls iris.hip.*, but the iris Python
#      package is not installed here. It is a pure-Python package and a checkout
#      exists at ~/amd-master/iris, so it is staged onto PYTHONPATH.
#   2. rank-1 also does a literal
#         open("/usr/local/lib/python3.10/dist-packages/iris/__init__.py")
#      whose failure is fatal. Staging the package at exactly that path
#      satisfies both the import and the literal read with one copy, instead of
#      patching their file.
#   3. rank-1 line 25 runs
#         os.system("sudo sed -i '66,82 s/^/#/' .../iris/__init__.py")
#      which was a workaround for whatever iris revision their environment had.
#      Line numbers 66-82 in the revision available here cover
#      `from . import hip`, `from . import experimental` and
#      `from .logging import ...`; commenting those out would remove the very
#      iris.hip that rank-1 then calls. The import is verified to work unpatched
#      below, so a no-op `sudo` shim is placed early on PATH to keep that sed
#      from firing. This is recorded rather than hidden.
#
# Root is needed only to write /usr/local/lib, hence a separate container.
set -uo pipefail

NAME=dhk-eval
IMAGE=vllm/vllm-openai-rocm:v0.26.0
IRIS_SRC=/home/subvadla/amd-master/iris/iris
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
STAGE=/usr/local/lib/python3.10/dist-packages

if [ "$(docker ps -aq -f name=^${NAME}$)" != "" ]; then docker rm -f "$NAME" >/dev/null; fi
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

docker run -d --name "$NAME" --entrypoint /bin/bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
  --ipc=host --shm-size 32G \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /home/subvadla:/home/subvadla \
  -w /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight \
  "$IMAGE" -c 'sleep infinity' >/dev/null
sleep 3
echo "container: $(docker ps -f name=^${NAME}$ --format '{{.Status}}')"

run() { docker exec "$NAME" bash -c "$1"; }

echo
echo "===== toolchain in $NAME ====="
run 'python3 -V; python3 -c "import torch;print(\"torch\",torch.__version__,\"gpus\",torch.cuda.device_count())"; python3 -c "import triton;print(\"triton\",triton.__version__)"; which sudo || echo "sudo: absent"'

echo
echo "===== stage the iris python package ====="
run "mkdir -p $STAGE && cp -r $IRIS_SRC $STAGE/ && ls $STAGE/iris | head -12 && wc -l $STAGE/iris/__init__.py"

echo
echo "===== no-op sudo shim (see header note 3) ====="
run 'mkdir -p /usr/local/shim && printf "#!/bin/sh\nexit 0\n" > /usr/local/shim/sudo && chmod +x /usr/local/shim/sudo && echo shim ready'

echo
echo "===== verify iris imports UNPATCHED and exposes what rank-1 calls ====="
run "PYTHONPATH=$STAGE python3 -c \"
import iris
print('iris ok:', iris.__file__)
print('iris.hip:', hasattr(iris, 'hip'))
print('hipIpcMemHandle_t:', hasattr(iris.hip, 'hipIpcMemHandle_t'))
print('get_ipc_handle:', hasattr(iris.hip, 'get_ipc_handle'))
print('Iris:', hasattr(iris, 'Iris'))
\""
if [ $? -ne 0 ]; then echo "iris import FAILED - stopping"; exit 1; fi

echo
echo "===== stage the evaluator arm ====="
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
DIR=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1
run "rm -rf $DIR && mkdir -p $DIR && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $DIR/ && cp $RANK1 $DIR/submission.py && cp $SRC/cases.txt $DIR/cases_test.txt && sha256sum $DIR/submission.py"
run "cat > $DIR/cases_bench.txt <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
wc -l $DIR/cases_bench.txt"

ENVS="PATH=/usr/local/shim:\$PATH PYTHONPATH=$STAGE POPCORN_FD=3 POPCORN_GPUS=8 \
TRITON_CACHE_DIR=$DIR/.triton TORCH_EXTENSIONS_DIR=$DIR/.ext TMPDIR=$DIR/.tmp"

for mode in test benchmark; do
  cases=cases_test.txt
  [ "$mode" = "benchmark" ] && cases=cases_bench.txt
  echo
  echo "################################################################"
  echo "# rank1 : eval.py $mode $cases"
  echo "################################################################"
  run "mkdir -p $DIR/.tmp && cd $DIR && $ENVS timeout 1500 python3 eval.py $mode $cases \
       3>$DIR/$mode.popcorn.txt >$DIR/$mode.stdout.txt 2>$DIR/$mode.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$mode.popcorn.txt 2>/dev/null | head -70"
  echo "--- stderr tail ---"
  run "tail -20 $DIR/$mode.stderr.txt 2>/dev/null"
done

echo
echo "===== DONE ====="
