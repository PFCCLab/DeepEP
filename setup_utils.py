# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import re
import subprocess


def _detect_local_gpu_arch():
    """Auto-detect the first visible GPU compute capability."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        )
        caps = {
            line.strip()
            for line in out.decode().splitlines()
            if line.strip()
        }
        return sorted(caps)[0] if caps else None
    except Exception:
        return None


def _cuda_arch_entries(cuda_arch_list: str) -> list[str]:
    entries = [
        part
        for part in re.split(r'[,\s;]+', cuda_arch_list.strip())
        if part
    ]
    if not entries:
        raise ValueError('PADDLE_CUDA_ARCH_LIST must not be empty')
    return entries


def _validate_sm90_or_newer(cuda_arch_list: str) -> None:
    for arch in _cuda_arch_entries(cuda_arch_list):
        match = re.fullmatch(r'(\d+)\.(\d+)([A-Za-z]?)', arch)
        if not match:
            raise ValueError(f'Invalid CUDA architecture format: {arch}')
        major = int(match.group(1))
        if major < 9:
            raise ValueError(
                'DeepEP requires GPU compute capability >= 9.0 (SM90+), '
                f'but got architecture {arch}. '
                'Please use a GPU with SM90+ or set PADDLE_CUDA_ARCH_LIST '
                'to a supported architecture.'
            )


def resolve_cuda_arch(default: str = '9.0') -> str:
    """Resolve compile architectures: env > local GPU auto-detect > default."""
    cuda_arch_list = os.getenv('PADDLE_CUDA_ARCH_LIST', '').strip()
    if not cuda_arch_list:
        cuda_arch_list = _detect_local_gpu_arch() or default
    _validate_sm90_or_newer(cuda_arch_list)
    return cuda_arch_list


def should_disable_aggressive_ptx(cuda_arch_list: str) -> bool:
    return any(
        arch not in ('9.0', '9.0a')
        for arch in _cuda_arch_entries(cuda_arch_list)
    )
