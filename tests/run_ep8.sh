#!/bin/bash
# Single-node 8-GPU launcher for the tests in this repo.
#
#   bash tests/run_ep8.sh tests/run_official_ep.py --deterministic --skip-perf-test
#   LOGTAG=ep8_det GPUS=0,1,2,3 bash tests/run_ep8.sh tests/order_vs_value.py --deterministic
#
# The first argument is the test script; the rest are forwarded to it.
set -u

DE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Only needed when NCCL does not live where Python would find it by default, e.g. when a newer
# NCCL is kept isolated from the one bundled with the Paddle wheel. Point this at the directory
# that *contains* `nvidia/nccl`, i.e. a `pip install --target` root.
NROOT=${EP_NCCL_PKG_ROOT:-}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
if [ $# -lt 1 ]; then
    echo "usage: bash tests/run_ep8.sh <test script> [args...]" >&2
    exit 1
fi
SCRIPT=$1
shift
LOGDIR=$DE/log/${LOGTAG:-$(basename "$SCRIPT" .py)}

rm -rf "$LOGDIR"

# The PaddleCloud pod exports a job-wide PADDLE_*/TRAINER_* topology that makes
# `paddle.distributed.launch` try to reach the other nodes. Strip it so launch builds a
# fresh single-node job.
# NOTES: `EP_NCCL_ROOT_DIR` comes from `deep_ep/envs.py` (baked in at build time), and
# `deep_ep.align_paddle_nccl_dir()` repoints Paddle's own `FLAGS_nccl_dir` at the same NCCL, so
# neither has to be set here and the `check_nccl_so()` guard can stay enabled.
cd "$DE" || exit 1
if [ -n "$NROOT" ]; then
    # `$NROOT` must come first on `PYTHONPATH`: torch resolves its NCCL package through `sys.path`,
    # and `find_nccl_root()` picks whichever `nvidia-nccl` distribution appears earliest there.
    EXTRA_PYTHONPATH="$NROOT:"
    EXTRA_LD_LIBRARY_PATH="$NROOT/nvidia/nccl/lib:"
else
    EXTRA_PYTHONPATH=""
    EXTRA_LD_LIBRARY_PATH=""
fi
env -u PADDLE_TRAINERS -u PADDLE_TRAINERS_NUM -u PADDLE_TRAINER_COUNT -u PADDLE_TRAINER_ID \
    -u PADDLE_CURRENT_ENDPOINT -u PADDLE_WORKERS_IP_PORT_LIST -u DISTRIBUTED_TRAINER_ENDPOINTS \
    -u PADDLE_TRAINER_ENDPOINTS -u PADDLE_PORT -u TRAINER_PORTS -u TRAINER_PORTS_NUM \
    -u PADDLE_PSERVERS_IP_PORT_LIST -u PADDLE_TRAINING_ROLE -u WORLD_SIZE -u RANK \
    -u MASTER_ADDR -u MASTER_PORT -u POD_INDEX \
    LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="${EXTRA_PYTHONPATH}$DE" \
    python -m paddle.distributed.launch \
        --log_dir "$LOGDIR" --gpus "$GPUS" --run_mode=collective \
        "$SCRIPT" "$@"

echo "===== $LOGDIR/workerlog.0 (tail) ====="
tail -n 60 "$LOGDIR/workerlog.0" 2>/dev/null
