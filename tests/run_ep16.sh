#!/bin/bash
# Two-node 16-GPU launcher for the tests in this repo.
#
#   bash tests/run_ep16.sh <nodeA> <nodeB> tests/run_official_ep.py --deterministic
#   NPROC_PER_NODE=8 bash tests/run_ep16.sh 10.0.0.1 10.0.0.2 tests/run_official_ep.py
#
# Run this from `<nodeA>`; the first two arguments are the node addresses, the third is the test
# script, and the rest are forwarded to it.
#
# There is no `paddle.distributed.launch` here to publish the `PADDLE_*` endpoints, so
# `tests/mpi_bootstrap.py` exchanges them through the rendezvous directory `EP_RV_DIR`, which must
# be visible from both nodes. `--map-by slot` is required: `--map-by node` interleaves ranks across
# the two hosts and breaks the EP scale-up / scale-out topology.
set -u

DE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# See `tests/run_ep8.sh` -- only needed when NCCL is kept outside the default location.
NROOT=${EP_NCCL_PKG_ROOT:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [ $# -lt 3 ]; then
    echo "usage: bash tests/run_ep16.sh <nodeA> <nodeB> <test script> [args...]" >&2
    exit 1
fi
NODE_A=$1
NODE_B=$2
SCRIPT=$3
shift 3

LOGDIR=$DE/log/${LOGTAG:-ep16_$(basename "$SCRIPT" .py)}
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR/rv"

if [ -n "$NROOT" ]; then
    EXTRA_PYTHONPATH="$NROOT:"
    EXTRA_LD_LIBRARY_PATH="$NROOT/nvidia/nccl/lib:"
else
    EXTRA_PYTHONPATH=""
    EXTRA_LD_LIBRARY_PATH=""
fi

cd "$DE" || exit 1
mpirun --allow-run-as-root \
    -host "$NODE_A:$NPROC_PER_NODE,$NODE_B:$NPROC_PER_NODE" \
    -np $((NPROC_PER_NODE * 2)) --map-by slot --bind-to none \
    --output-filename "$LOGDIR/mpi" \
    -x PATH \
    -x LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:-}" \
    -x PYTHONPATH="${EXTRA_PYTHONPATH}$DE" \
    -x HOME="$DE" \
    -x CUDA_VISIBLE_DEVICES \
    -x EP_RV_DIR="$LOGDIR/rv" \
    ${EP_RELAXED_WIN_ORDERING:+-x EP_RELAXED_WIN_ORDERING="$EP_RELAXED_WIN_ORDERING"} \
    ${EP_OVERRIDE_RDMA_SL:+-x EP_OVERRIDE_RDMA_SL="$EP_OVERRIDE_RDMA_SL"} \
    ${EP_RDMA_GBS:+-x EP_RDMA_GBS="$EP_RDMA_GBS"} \    -x FLAGS_use_virtual_memory_auto_growth=1 \
    -x FLAGS_tcp_store_using_libuv=0 \
    -wdir "$DE" \
    python -u "$SCRIPT" "$@"
echo "mpirun exit=$?"

# NOTES: mpirun writes per-rank files rather than the `workerlog.N` that `launch` produces, and the
# rank directories are zero-padded (`rank.00`).
echo "===== $LOGDIR/mpi/1/rank.00/stdout (tail) ====="
tail -n 40 "$LOGDIR/mpi/1/rank.00/stdout" 2>/dev/null
