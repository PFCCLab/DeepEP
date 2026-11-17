# Copyright (c) 2026 PaddlePaddle CORPORATION & AFFILIATES. All rights reserved.
import ctypes
import ctypes.util
import importlib.util
import os
import platform
import sysconfig
from pathlib import Path
from typing import NamedTuple


class RuntimePaths(NamedTuple):
    cuda_home: str | None
    rdma_include_dir: str | None
    rdma_library_dir: str | None


def _multiarch_triplets() -> list[str]:
    triplets = []
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        triplets.append(multiarch)

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        triplets.append("x86_64-linux-gnu")
    elif machine in {"aarch64", "arm64"}:
        triplets.append("aarch64-linux-gnu")

    return list(dict.fromkeys(triplets))


def _rdma_lib_dirs(prefix: str) -> list[Path]:
    root = Path(prefix)
    dirs = [root / "lib", root / "lib64"]
    dirs.extend(root / "lib" / triplet for triplet in _multiarch_triplets())
    return dirs


def _has_rdma_runtime_libs(lib_dir: Path) -> bool:
    return any(lib_dir.glob("libibverbs.so*")) and any(lib_dir.glob("libmlx5.so*"))


def _first_env_path(*names: str) -> Path | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return Path(value)
    return None


def _has_cuda_toolkit(root: Path) -> bool:
    return (root / "bin" / "nvcc").exists() and (root / "include" / "cuda_runtime.h").exists()


def _find_cuda_toolkit_home() -> str | None:
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.getenv(env_name)
        if cuda_home and _has_cuda_toolkit(Path(cuda_home)):
            return cuda_home

    try:
        from paddle.utils.cpp_extension.extension_utils import find_cuda_home

        cuda_home = find_cuda_home()
        if cuda_home and _has_cuda_toolkit(Path(cuda_home)):
            return cuda_home
    except Exception:
        pass

    for candidate in ("/usr/local/cuda", "/opt/cuda"):
        if _has_cuda_toolkit(Path(candidate)):
            return candidate
    return None


def detect_cuda_home() -> str | None:
    explicit_cuda_home = _first_env_path("CUDA_HOME", "CUDA_PATH")
    if explicit_cuda_home:
        return str(explicit_cuda_home)

    return _find_cuda_toolkit_home()


def _find_rdma_include_dir(prefix: str | None) -> Path | None:
    candidates = []
    env_include_dir = os.getenv("RDMA_CORE_INCLUDE_DIR")
    if env_include_dir:
        candidates.append(Path(env_include_dir))
    if prefix:
        candidates.append(Path(prefix) / "include")

    for include_dir in candidates:
        if (include_dir / "infiniband" / "verbs.h").exists():
            return include_dir
    return None


def _find_rdma_library_dir(prefix: str | None) -> Path | None:
    candidates = []
    env_library_dir = os.getenv("RDMA_CORE_LIBRARY_DIR")
    if env_library_dir:
        candidates.append(Path(env_library_dir))
    if prefix:
        candidates.extend(_rdma_lib_dirs(prefix))
    for env_name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        candidates.extend(Path(path) for path in os.getenv(env_name, "").split(os.pathsep) if path)

    for lib_dir in dict.fromkeys(candidates):
        if _has_rdma_runtime_libs(lib_dir):
            return lib_dir
    return None


def detect_rdma_core_paths() -> tuple[Path | None, Path | None]:
    rdma_core_home = os.getenv("RDMA_CORE_HOME")
    prefixes = [rdma_core_home] if rdma_core_home else ["/usr/local", "/usr"]
    include_dir = None
    library_dir = None
    for prefix in prefixes:
        if include_dir is None:
            include_dir = _find_rdma_include_dir(prefix)
        if library_dir is None:
            library_dir = _find_rdma_library_dir(prefix)
        if include_dir and library_dir:
            break

    return include_dir, library_dir


def _preload_rdma_libraries(lib_dir: Path | None) -> None:
    lib_names = ("ibverbs", "mlx5")
    for lib_name in lib_names:
        candidates = []
        if lib_dir:
            candidates.extend(sorted(lib_dir.glob(f"lib{lib_name}.so*")))
        found = ctypes.util.find_library(lib_name)
        if found:
            candidates.append(Path(found))

        for candidate in candidates:
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


def _preload_first(candidates: list[Path | str]) -> None:
    for candidate in candidates:
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _preload_paddle_libraries() -> None:
    try:
        import paddle

        paddle_dir = Path(paddle.__file__).resolve().parent
        _preload_first([paddle_dir / "libs" / "libpaddle.so"])
    except Exception:
        return


def _preload_driver_libraries() -> None:
    for lib_name in ("cuda", "nvidia-ml"):
        found = ctypes.util.find_library(lib_name)
        if found:
            _preload_first([found])


def _preload_nvshmem_libraries() -> None:
    spec = importlib.util.find_spec("nvidia.nvshmem")
    if not spec or not spec.submodule_search_locations:
        return
    nvshmem_dir = Path(next(iter(spec.submodule_search_locations)))
    _preload_first(sorted((nvshmem_dir / "lib").glob("libnvshmem_host.so*")))


def detect_runtime_paths() -> RuntimePaths:
    rdma_include_dir, rdma_library_dir = detect_rdma_core_paths()
    return RuntimePaths(
        cuda_home=detect_cuda_home(),
        rdma_include_dir=str(rdma_include_dir) if rdma_include_dir else None,
        rdma_library_dir=str(rdma_library_dir) if rdma_library_dir else None,
    )


def configure_runtime_paths() -> RuntimePaths:
    _preload_paddle_libraries()
    runtime_paths = detect_runtime_paths()
    _preload_driver_libraries()
    _preload_nvshmem_libraries()
    rdma_library_dir = Path(runtime_paths.rdma_library_dir) if runtime_paths.rdma_library_dir else None
    _preload_rdma_libraries(rdma_library_dir)
    return runtime_paths
