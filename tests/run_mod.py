"""Generic launcher: run a v2.5 test module (that uses init_dist + torch.multiprocessing.spawn)
under Paddle via launch/mpirun + fleet, one rank per process.

Usage: python -m paddle.distributed.launch ... tests/run_mod.py <test_file> [-- test args...]
"""
import os
import sys
import runpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MPI = 'OMPI_COMM_WORLD_RANK' in os.environ
if _MPI:
    from tests.mpi_bootstrap import setup_paddle_env
    setup_paddle_env()

_LOCAL_RANK = int(os.environ.get('PADDLE_LOCAL_RANK', os.environ.get('FLAGS_selected_gpus', '0')) or '0')

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
paddle.enable_compat()
import torch  # noqa


def _fleet_init_dist(local_rank, num_local_ranks, seed=0):
    total = num_local_ranks if _MPI else num_local_ranks * int(os.getenv('WORLD_SIZE', '1'))
    group = dist.new_group(list(range(total)))
    import deep_ep.utils.envs as _envs
    _envs._local_rank = local_rank
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)
    _envs.init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), group


def main():
    test_file = sys.argv[1]
    test_args = sys.argv[2:]

    world = int(os.environ.get('PADDLE_TRAINERS_NUM', '1'))
    if world > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {'dp_degree': 1, 'mp_degree': world, 'pp_degree': 1, 'sharding_degree': 1}
        fleet.init(is_collective=True, strategy=strategy)

    # Patch deep_ep's init_dist to the fleet version
    import deep_ep.utils.envs as _envs
    _envs.init_dist = _fleet_init_dist

    # Patch spawn to run a single rank in-process (launch already gives 1 proc/rank).
    # NOTE(paddle-compat): torch.multiprocessing does not exist under Paddle; inject a stub.
    import types
    def _fake_spawn(fn, args=(), nprocs=1, **kw):
        fn(_LOCAL_RANK, *args)
    _fake_mp = types.ModuleType('torch.multiprocessing')
    _fake_mp.spawn = _fake_spawn
    torch.multiprocessing = _fake_mp
    sys.modules['torch.multiprocessing'] = _fake_mp

    # NOTE(paddle-compat): shim a few torch APIs the tests rely on that paddle's
    # torch-compat layer does not provide yet (test-harness only, not DeepEP code).
    #  - torch.cuda._sleep: GPU busy-wait used only for timing/pipelining realism.
    #  - torch.randint/torch.rand: paddle ignores the `generator=` kwarg.
    #  - torch.distributed.all_gather_single: == torch's all_gather_into_tensor.
    if not hasattr(torch.cuda, '_sleep'):
        torch.cuda._sleep = lambda _cycles: None
    _orig_randint, _orig_rand = torch.randint, torch.rand
    def _seed_from(generator):
        # NOTE(paddle-compat): paddle's randint/rand take no `generator=`; honor it by
        # seeding paddle's global RNG so results stay bit-identical across ranks (the
        # redundancy-mapping tests rely on every rank drawing the same table).
        if generator is not None:
            paddle.seed(int(generator.initial_seed()))
    def _randint(*a, generator=None, **kw):
        _seed_from(generator)
        return _orig_randint(*a, **kw)
    def _rand(*a, generator=None, **kw):
        _seed_from(generator)
        return _orig_rand(*a, **kw)
    torch.randint, torch.rand = _randint, _rand
    # NOTE(paddle-compat): paddle's `divide(..., rounding_mode='floor')` needs a Tensor divisor.
    _orig_div = torch.div
    def _div(x, y, *, rounding_mode=None, **kw):
        if rounding_mode is not None and not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return _orig_div(x, y, rounding_mode=rounding_mode, **kw)
    torch.div = _div
    def _all_gather_single(output, input, group=None):
        ws = group.world_size if group is not None else dist.get_world_size()
        chunks = [torch.empty_like(input) for _ in range(ws)]
        dist.all_gather(chunks, input, group=group)
        output.copy_(torch.cat(chunks, dim=0).reshape(output.shape))
    if not hasattr(torch.distributed, 'all_gather_single'):
        torch.distributed.all_gather_single = _all_gather_single

    # Run the target test module as __main__ with its own argv
    sys.argv = [test_file] + test_args
    runpy.run_path(test_file, run_name='__main__')
    if dist.get_rank() == 0:
        print(f'\n[rank0] MOD {os.path.basename(test_file)} DONE')


if __name__ == '__main__':
    main()
