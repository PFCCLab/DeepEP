# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import platform
import subprocess
import setuptools
import importlib
import shutil
import re
import sysconfig

from pathlib import Path
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, _get_cuda_arch_flags
from paddle.utils.cpp_extension.extension_utils import (
    add_compile_flag,
)
from setup_utils import resolve_cuda_arch

def collect_package_files(package: str, relative_dir: str):
    base_path = Path(package) / relative_dir
    if not base_path.exists():
        return []
    return [
        str(path.relative_to(package))
        for path in base_path.rglob('*')
        if path.is_file()
    ]


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')

def to_nvcc_gencode(s: str) -> str:
    flags = []
    for part in re.split(r'[,\s;]+', s.strip()):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\.(\d+)([A-Za-z]?)', part)
        if not m:
            raise ValueError(f"Invalid entry: {part}")
        major, minor, suf = m.groups()
        arch = f"{int(major)}{int(minor)}{suf.lower()}"
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return " ".join(flags)


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


def _rdma_lib_dirs(prefix: str) -> list[str]:
    dirs = [os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")]
    dirs.extend(os.path.join(prefix, "lib", triplet) for triplet in _multiarch_triplets())
    return dirs


def _has_rdma_libs(lib_dir: str) -> bool:
    return os.path.exists(os.path.join(lib_dir, "libibverbs.so")) and os.path.exists(
        os.path.join(lib_dir, "libmlx5.so")
    )


def _find_rdma_include_dir(prefix: str | None) -> str | None:
    candidates = []
    env_include_dir = os.getenv("RDMA_CORE_INCLUDE_DIR")
    if env_include_dir:
        candidates.append(env_include_dir)
    if prefix:
        candidates.append(os.path.join(prefix, "include"))

    for include_dir in candidates:
        if os.path.exists(os.path.join(include_dir, "infiniband", "verbs.h")):
            return include_dir
    return None


def _find_rdma_library_dir(prefix: str | None) -> str | None:
    candidates = []
    env_library_dir = os.getenv("RDMA_CORE_LIBRARY_DIR")
    if env_library_dir:
        candidates.append(env_library_dir)
    if prefix:
        candidates.extend(_rdma_lib_dirs(prefix))
    for env_name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        candidates.extend(path for path in os.getenv(env_name, "").split(os.pathsep) if path)

    for lib_dir in dict.fromkeys(candidates):
        if _has_rdma_libs(lib_dir):
            return lib_dir
    return None


def find_rdma_core_paths() -> tuple[str, str]:
    rdma_core_home = os.getenv("RDMA_CORE_HOME")
    prefixes = [rdma_core_home] if rdma_core_home else ["/usr/local", "/usr"]
    for prefix in prefixes:
        include_dir = _find_rdma_include_dir(prefix)
        library_dir = _find_rdma_library_dir(prefix)
        if include_dir and library_dir:
            return include_dir, library_dir
    raise RuntimeError(
        "Could not locate RDMA core headers/libs. Set RDMA_CORE_HOME or "
        "RDMA_CORE_INCLUDE_DIR/RDMA_CORE_LIBRARY_DIR."
    )


def ensure_multinode_jit_objects(extra_objects: list[str], current_dir: str) -> None:
    static_lib = os.path.join(current_dir, "third-party/nccl/build/lib/libnccl_static.a")
    obj_dir = os.path.join(current_dir, "deep_ep/backend/nccl/obj")
    missing = [os.path.basename(obj_path) for obj_path in extra_objects if not os.path.exists(obj_path)]
    if not missing:
        return
    if not os.path.exists(static_lib):
        raise FileNotFoundError(f"Missing NCCL static library for HybridEP JIT objects: {static_lib}")

    os.makedirs(obj_dir, exist_ok=True)
    subprocess.run(["ar", "x", static_lib, *missing], cwd=obj_dir, check=True)


def get_extension_hybrid_ep_cpp():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    enable_multinode = os.getenv("HYBRID_EP_MULTINODE", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    os.environ['PADDLE_CUDA_ARCH_LIST'] = resolve_cuda_arch(default='10.0')

    # Basic compile arguments
    compile_args = {
        "nvcc": [
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "--expt-relaxed-constexpr",
            "-O3",
            "--shared",
        ],
    }

    sources = [
        "csrc/hybrid_ep/hybrid_ep.cu",
        "csrc/hybrid_ep/allocator/allocator.cu",
        "csrc/hybrid_ep/jit/compiler.cu",
        "csrc/hybrid_ep/executor/executor.cu",
        "csrc/hybrid_ep/extension/permute.cu",
        "csrc/hybrid_ep/extension/allgather.cu",
        "csrc/hybrid_ep/pybind_hybrid_ep.cu",
    ]
    include_dirs = [
        os.path.join(current_dir, "csrc/hybrid_ep/"),
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
    ]
    library_dirs = []
    libraries = ["cuda", "nvtx3interop"]
    extra_objects = []
    runtime_library_dirs = []
    nvcc_dlink = ['-dlink']
    extra_link_args = ["-lcuda", "-lcudadevrt"]

    if len(nvcc_dlink) > 0:
        nvcc_dlink = nvcc_dlink + _get_cuda_arch_flags()
        compile_args['nvcc_dlink'] = nvcc_dlink


    # Add dependency for jit
    compile_args["nvcc"].extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
    compile_args["nvcc"].append(f'-DSM_ARCH="{os.environ["PADDLE_CUDA_ARCH_LIST"]}"')
    # Copy the hybrid backend code to python package for JIT compilation
    shutil.copytree(
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
        os.path.join(current_dir, "deep_ep/backend/"),
        dirs_exist_ok=True
    )
    # Add inter-node dependency 
    if enable_multinode:
        sources.extend(["csrc/hybrid_ep/internode.cu"])
        rdma_include_dir, rdma_library_dir = find_rdma_core_paths()
        nccl_dir = os.path.join(current_dir, "third-party/nccl")        
        compile_args["nvcc"].append("-DHYBRID_EP_BUILD_MULTINODE_ENABLE")
        extra_link_args.append(f"-l:libnvidia-ml.so.1")

        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=current_dir, check=True)
        # Generate the inter-node dependency to the python package for JIT compilation
        subprocess.run(["make", "-j", "src.build", f"NVCC_GENCODE={to_nvcc_gencode(os.environ['PADDLE_CUDA_ARCH_LIST'])}"], cwd=nccl_dir, check=True)
        # Add third-party dependency 
        include_dirs.append(os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"))
        include_dirs.append(rdma_include_dir)
        library_dirs.append(rdma_library_dir)
        libraries.append("mlx5")
        libraries.append("ibverbs")
        # Copy the inter-node dependency to python package
        shutil.copytree(
            os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"),
            os.path.join(current_dir, "deep_ep/backend/nccl/include"),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(nccl_dir, "build/obj/transport/net_ib/gdaki/doca-gpunetio"),
            os.path.join(current_dir, "deep_ep/backend/nccl/obj"),
            dirs_exist_ok=True
        )
        # Set the extra objects
        DOCA_OBJ_PATH = os.path.join(current_dir, "deep_ep/backend/nccl/obj")
        extra_objects = [
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_high_level.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_cuda_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_device_attr.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_ibv_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_mlx5dv_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_qp.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_cq.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_srq.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_uar.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_umem.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_gdrcopy.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_log.o"),
        ]
        ensure_multinode_jit_objects(extra_objects, current_dir)


    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {libraries}')
    print(f' > Library dirs: {library_dirs}')
    print(f' > Extra link args: {extra_link_args}')
    print(f' > Compilation flags: {compile_args}')
    print(f' > Extra objects: {extra_objects}')
    print(f' > Runtime library dirs: {runtime_library_dirs}')
    print(f' > Arch list: {os.environ["PADDLE_CUDA_ARCH_LIST"]}')
    print()

    add_compile_flag(compile_args, ['-DPADDLE_WITH_CUDA'])
    add_compile_flag(compile_args, ['-DWITH_DISTRIBUTE'])
    add_compile_flag(compile_args, ['-DWITH_NVSHMEM'])
    add_compile_flag(compile_args, ['-DWITH_GPU'])
    add_compile_flag(compile_args, ['-DWITH_FLUID_ONLY'])

    extension_hybrid_ep_cpp = CUDAExtension(
        name="hybrid_ep_cpp",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_objects=extra_objects,
        runtime_library_dirs=runtime_library_dirs,
        extra_link_args=extra_link_args,
    )

    return extension_hybrid_ep_cpp

if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.2.1' + revision,
        packages=setuptools.find_packages(
            include=['deep_ep']
        ),
        install_requires=[
            'pynvml',
        ],
        ext_modules=[
            get_extension_hybrid_ep_cpp(),
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        package_data={
            'deep_ep': collect_package_files('deep_ep', 'backend'),
        },
        include_package_data=True
    )
