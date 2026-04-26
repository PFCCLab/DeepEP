# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import subprocess
import shutil
from pathlib import Path

import setuptools
from paddle.utils.cpp_extension import BuildExtension
from setuptools.command.build_ext import build_ext

from setup_deep_ep import collect_package_files, get_extension_deep_ep_cpp
from setup_hybrid_ep import get_extension_hybrid_ep_cpp


def remove_runpath(shared_library: str) -> None:
    """Keep built extensions relocatable; runtime deps are preloaded in Python."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        return
    path = Path(shared_library)
    candidates = [path]
    if path.suffix:
        candidates.append(path.with_name(f"{path.stem}_pd_{path.suffix}"))
    for candidate in candidates:
        if candidate.exists():
            subprocess.run([patchelf, "--remove-rpath", str(candidate)], check=True)


class MultiExtensionBuild(BuildExtension):
    """Run Paddle's single-extension build path once per extension."""

    def run(self):
        build_ext.run(self)
        self._run_for_each_extension(BuildExtension._generate_python_api_file)
        if self.inplace:
            self._run_for_each_extension(BuildExtension._rename_inplace_shared_library)
        self._run_for_each_extension(self._remove_runpath_for_current_extension)

    def build_extensions(self):
        original_extensions = list(self.extensions)
        try:
            for extension in original_extensions:
                self.extensions = [extension]
                self.contain_cuda_file = False
                super().build_extensions()
        finally:
            self.extensions = original_extensions

    def _run_for_each_extension(self, func):
        original_extensions = list(self.extensions)
        try:
            for extension in original_extensions:
                self.extensions = [extension]
                func(self)
        finally:
            self.extensions = original_extensions

    def _remove_runpath_for_current_extension(self, *_):
        remove_runpath(self.get_ext_fullpath(self.extensions[0].name))


def get_revision() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except Exception:
        return ""


if __name__ == "__main__":
    ext_modules = [
        get_extension_deep_ep_cpp(),
        get_extension_hybrid_ep_cpp(),
    ]

    setuptools.setup(
        name="deep_ep",
        version="1.2.1" + get_revision(),
        packages=setuptools.find_packages(include=["deep_ep"]),
        install_requires=[
            "pynvml",
        ],
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": MultiExtensionBuild,
        },
        package_data={
            "deep_ep": collect_package_files("deep_ep", "backend"),
        },
        include_package_data=True,
    )
