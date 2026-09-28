#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import hashlib
from pathlib import Path
os.path.dirname(os.path.abspath(__file__))

source_root = Path(__file__).resolve().parent
build_sources = sorted(
    list((source_root / 'cuda_rasterizer').glob('*.h')) +
    list((source_root / 'cuda_rasterizer').glob('*.cu')) +
    [source_root / name for name in ('ext.cpp', 'rasterize_points.cu', 'rasterize_points.h')])
source_digest = hashlib.sha256(b''.join(
    str(path.relative_to(source_root)).encode() + b'\0' + path.read_bytes()
    for path in build_sources)).hexdigest()

setup(
    name="diff_surfel_rasterization",
    packages=['diff_surfel_rasterization'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": ['-DAABB_BACKWARD_SOURCE_SHA256="' + source_digest + '"'],
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
