#pragma once

// The projected center in compute_aabb is a quotient of two quadratic forms.
// Keep its actual VJP testable on the CPU as well as callable from the CUDA kernel.
#ifdef __CUDACC__
#define AABB_HOST_DEVICE __host__ __device__
#else
#define AABB_HOST_DEVICE
#endif

template <typename Scalar>
AABB_HOST_DEVICE inline void aabb_center_vjp(
    const Scalar* matrix, Scalar cutoff_squared, Scalar grad_x, Scalar grad_y,
    Scalar* gradient)
{
    const Scalar* u = matrix;
    const Scalar* v = matrix + 3;
    const Scalar* w = matrix + 6;
    const Scalar q[3] = {cutoff_squared, cutoff_squared, Scalar(-1)};
    const Scalar denominator = q[0]*w[0]*w[0] + q[1]*w[1]*w[1] + q[2]*w[2]*w[2];
    const Scalar px = (q[0]*u[0]*w[0] + q[1]*u[1]*w[1] + q[2]*u[2]*w[2]) / denominator;
    const Scalar py = (q[0]*v[0]*w[0] + q[1]*v[1]*w[1] + q[2]*v[2]*w[2]) / denominator;
    for (int j = 0; j < 3; ++j) {
        const Scalar factor = q[j] / denominator;
        gradient[j] += grad_x * factor * w[j];
        gradient[3+j] += grad_y * factor * w[j];
        gradient[6+j] += factor * (
            grad_x * (u[j] - Scalar(2)*px*w[j]) +
            grad_y * (v[j] - Scalar(2)*py*w[j]));
    }
}

#undef AABB_HOST_DEVICE
