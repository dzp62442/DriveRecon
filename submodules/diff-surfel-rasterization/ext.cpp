/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#ifndef AABB_BACKWARD_SOURCE_SHA256
#define AABB_BACKWARD_SOURCE_SHA256 "unknown"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("aabb_backward_version") = 1;
  m.attr("aabb_backward_source_sha256") = AABB_BACKWARD_SOURCE_SHA256;
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
