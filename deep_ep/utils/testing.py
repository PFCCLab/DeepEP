import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist


def flush_l2_cache(enabled: bool = True):
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6
    if enabled:
        torch.empty(int(l2_flush_cache_size // 4), dtype=torch.int, device='cuda').zero_()


def bench(fn, num_warmups: int = 30, num_tests: int = 30,
          post_fn: Optional[Callable] = None, flush_l2: bool = True):
    """
    Benchmark a function using CUDA events.

    Arguments:
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.

    Returns:
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        flush_l2_cache(flush_l2)
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    """
    Context manager to suppress stdout and stderr output.
    """

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 flush_l2: bool = True,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1,
                 barrier: Optional[Callable] = None):
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.

    Arguments:
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        suppress_kineto_output: whether to suppress profiler output.
        trace_path: the path to save the Chrome trace (`None` to skip).
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier_comm_profiling: whether to insert a barrier before each iteration to reduce
            unbalanced CPU launch overhead.
        num_kernels_per_period: the number of kernels launched per test period.
        barrier: a custom barrier function to use instead of `dist.all_reduce`.

    Returns:
        durations: the average kernel duration(s) in seconds.
    """
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get('EP_USE_NVIDIA_TOOLS', 0)):
        return (1, ) * len(kernel_names) if is_tuple else 1

    # For some auto-tuning kernels with prints
    fn()
    torch.cuda.synchronize()

    # Profile with paddle.profiler (CUPTI-backed) and derive per-kernel GPU time by name.
    # NOTE(paddle-compat): Paddle has no torch.profiler.{schedule,profile,ProfilerActivity} nor
    # key_averages(). Use Profiler + make_scheduler + export_chrome_tracing, then parse device
    # kernel events (cat == 'Kernel') from the emitted chrome trace.
    import paddle
    from paddle import profiler as _P
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0
    _trace_dir = tempfile.mkdtemp(prefix='ep_kineto_')
    with suppress():
        # Equivalent to torch schedule(wait=0, warmup=1, active=1, repeat=1): skip cycle 1 (warmup),
        # record cycle 2.
        scheduler = _P.make_scheduler(closed=0, ready=1, record=1, repeat=1)
        profiler = _P.Profiler(targets=[_P.ProfilerTarget.GPU], scheduler=scheduler,
                               on_trace_ready=_P.export_chrome_tracing(_trace_dir))
        dummy = torch.ones(1, dtype=torch.float, device='cuda')
        profiler.start()
        for i in range(2):
            for _ in range(num_tests):
                # Flush L2 cache
                flush_l2_cache(flush_l2)

                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    # NOTE(paddle-compat): torch.cuda._sleep (a lightweight ~10ms GPU busy-wait) has
                    # no Paddle equivalent, so skip it when absent. It only smooths launch jitter;
                    # skipping adds timing noise but not incorrectness. Do NOT substitute a heavy op
                    # (e.g. a large matmul): it would contend for the GPU and pollute the measurement.
                    _sleep = getattr(torch.cuda, '_sleep', None)
                    if _sleep is not None:
                        _sleep(int(2e7)) # ~10ms

                    # Some network may have ring-based implement, so be careful to use `all_reduce`
                    if barrier is None:
                        dist.all_reduce(dummy)
                    else:
                        barrier()
                fn()
            torch.cuda.synchronize()
            profiler.step()
        profiler.stop()

    # Parse the chrome trace: collect device-kernel events (cat == 'Kernel') by name
    import glob
    trace_files = sorted(glob.glob(os.path.join(_trace_dir, '*.json')), key=os.path.getmtime)
    assert trace_files, f'paddle profiler produced no trace in {_trace_dir}'
    trace_data = json.loads(Path(trace_files[-1]).read_text())
    all_events = trace_data['traceEvents'] if isinstance(trace_data, dict) else trace_data
    kernel_events = [e for e in all_events
                     if isinstance(e, dict) and e.get('cat') == 'Kernel' and 'dur' in e and 'name' in e]

    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])

    # Save chrome trace
    if trace_path is not None:
        import shutil
        shutil.copy(trace_files[-1], trace_path)

    # Return average kernel durations (seconds). `dur` in paddle chrome trace is microseconds.
    kernel_durations = []
    for name in kernel_names:
        matched = sorted([e for e in kernel_events if name in e['name']], key=lambda e: e['ts'])
        durations = [e['dur'] / 1e6 for e in matched]
        if int(os.environ.get('EP_BENCH_DEBUG', 0)):
            _us = [round(d * 1e6, 1) for d in durations]
            print(f'[bench_kineto] kernel={name!r} matched={len(durations)} '
                  f'avg={sum(_us)/len(_us) if _us else 0:.1f}us durs(us)={_us[:40]}', flush=True)
        if num_kernels_per_period > 1:
            assert len(durations) % num_kernels_per_period == 0, \
                f'{name}: got {len(durations)} kernels, not divisible by {num_kernels_per_period}'
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations.append([sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                                     for j in range(num_kernels_per_period)])
        else:
            kernel_durations.append(sum(durations) / len(durations) if durations else 0.0)

    # Clean up the temporary trace directory
    import shutil
    shutil.rmtree(_trace_dir, ignore_errors=True)

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def parse_num_bytes(text: str) -> int:
    """Parse a byte count, optionally with a binary suffix: `1G`, `64M`, `512k`.
    """
    match = re.fullmatch(r'\s*([0-9.]+)\s*([kmgt])?(?:i?b)?\s*', text, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f'{text!r} is not a byte count; expected e.g. 1G, 64M, 512K, 1048576')
    scale = {None: 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}
    suffix = match.group(2)
    try:
        value = int(float(match.group(1)) * scale[suffix.lower() if suffix else None])
    except ValueError:
        raise ValueError(f'{text!r} has an unparseable number')
    if value <= 0:
        raise ValueError(f'byte count must be positive, got {value}')
    return value
