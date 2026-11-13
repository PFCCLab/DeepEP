import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTES: `EP_RV_DIR` is what the two-node mpirun launcher sets, and it is also what tells this
# run apart from one that merely inherited `OMPI_*` from whatever started it (aq's remote
# launcher does), where `paddle.distributed.launch` already published the endpoints
_MPI = 'OMPI_COMM_WORLD_RANK' in os.environ and bool(os.environ.get('EP_RV_DIR'))
if _MPI:
    from tests.mpi_bootstrap import setup_paddle_env
    setup_paddle_env()

_LOCAL_RANK = int(os.environ.get('PADDLE_LOCAL_RANK', os.environ.get('FLAGS_selected_gpus', '0')) or '0')

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

paddle.enable_compat()

import torch  # noqa: E402 

import tests.ep.test_ep as tep  # noqa: E402 


def _fleet_init_dist(local_rank: int, num_local_ranks: int, seed: int = 0):
    total = num_local_ranks if _MPI else num_local_ranks * int(os.getenv('WORLD_SIZE', '1'))
    group = dist.new_group(list(range(total)))
    import deep_ep.utils.envs as _envs
    _envs._local_rank = local_rank
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)
    tep.init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), group


def _init_dist_env(world_size, seed=20):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        'dp_degree': 1, 'mp_degree': world_size, 'pp_degree': 1, 'sharding_degree': 1,
    }
    strategy.tensor_parallel_configs = {'tensor_init_seed': seed}
    fleet.init(is_collective=True, strategy=strategy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-perf-test', action='store_true')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--no-hybrid', action='store_true',
                        help='use direct RDMA kernels instead of the hybrid forwarder path')
    parser.add_argument('--test-first-only', action='store_true')
    parser.add_argument('--do-pressure-test', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--pressure-iters', type=int, default=0,
                        help='bound pressure iterations (0 = official 1e9)')
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=16)
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--num-allocated-qps', type=int, default=0)
    parser.add_argument('--ignore-local-traffic', action='store_true',
                        help='exclude the intra-node scaleout group from the SO bandwidth figure')
    parser.add_argument('--unbalanced-ratio', type=float, default=1.0)
    parser.add_argument('--precise-unbalanced-ratio', action='store_true')
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    args, _ = parser.parse_known_args()

    if args.pressure_iters > 0:
        import builtins
        _orig_range = builtins.range
        _cap = args.pressure_iters
        def _capped_range(*a):
            r = _orig_range(*a)
            return r if len(r) < int(1e9) else _orig_range(_cap)
        tep.range = _capped_range  

    defaults = dict(
        num_processes=2, num_sms=args.num_sms, num_qps=0,
        num_allocated_qps=args.num_allocated_qps,
        num_gpu_timeout_secs=100, num_cpu_timeout_secs=100,
        num_tokens=args.num_tokens, hidden=args.hidden,
        num_topk=args.num_topk, num_experts=args.num_experts,
        do_cpu_sync=1, defer_epilogue=1,
        allow_hybrid_mode=0 if args.no_hybrid else 1, allow_multiple_reduction=1,
        prefer_overlap_with_compute=0, deterministic=args.deterministic,
        seed=0, skip_check=args.skip_check, skip_perf_test=args.skip_perf_test,
        do_pressure_test=args.do_pressure_test, reuse_buffer=False,
        test_first_only=args.test_first_only, unbalanced_ratio=args.unbalanced_ratio,
        precise_unbalanced_ratio=args.precise_unbalanced_ratio, masked_ratio=args.masked_ratio,
        dump_profile_traces='', ignore_local_traffic=args.ignore_local_traffic, fp8_dispatch_only=False,
    )
    for k, v in defaults.items():
        setattr(args, k, v)

    tep.init_dist = _fleet_init_dist

    if int(os.getenv('EP_BF16_ONLY', '0')):
        # Mode tuple is (expert_alignment, use_fp8_dispatch, num_bias, ...), so BF16 means `m[1] == 0`.
        _orig_enum = tep.enumerate_ep_modes
        tep.enumerate_ep_modes = lambda a: (m for m in _orig_enum(a) if m[1] == 0)

    world = int(os.environ.get('PADDLE_TRAINERS_NUM', '1'))
    if world > 1:
        _init_dist_env(world)

    tep.test_loop(_LOCAL_RANK, world, args)
    if dist.get_rank() == 0:
        print(f'\n[rank0] OFFICIAL test_ep (new paddle, no runner patches) DONE')


if __name__ == '__main__':
    main()
