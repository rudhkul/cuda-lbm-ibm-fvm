
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::cerr << "CUDA failure at " << __FILE__ << ":" << __LINE__               \
                      << " : " << cudaGetErrorString(err__) << std::endl;                \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

struct Params {
    int nx, ny, nz;
    int imin, imax;
    int jmin, jmax;
    int kmin, kmax;
    float Lx, Ly, Lz;
    float rho_l, rho_g;
    float nu_l, nu_g;
    float sigma;
    float gx, gy, gz;
    float dt;
    float dx, dy, dz;
    float dxi, dyi, dzi;
    int nTot;
};

__constant__ Params cpar;
Params cpar_host;

static volatile std::sig_atomic_t g_stop = 0;

void handle_signal(int) {
    g_stop = 1;
}

__host__ __device__ inline float median3f(const float a, const float b, const float c) {
    const float mx = fmaxf(fmaxf(a, b), c);
    const float mn = fminf(fminf(a, b), c);
    return a + b + c - mx - mn;
}

__host__ __device__ inline float clamp01f(const float v) {
    return median3f(0.0f, 1.0f, v);
}

__host__ __device__ inline int clampi(const int v, const int lo, const int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline size_t idx3d(const int i, const int j, const int k, const int nxTot, const int nyTot) {
    return (static_cast<size_t>(k) * static_cast<size_t>(nyTot) + static_cast<size_t>(j)) *
               static_cast<size_t>(nxTot) +
           static_cast<size_t>(i);
}

__global__ void set_init_F_kernel(float* __restrict__ F, const int ic) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;

    if (i >= nxTot || j >= nyTot || k >= nzTot) return;

    if (ic == 1) {
        const float x = static_cast<float>(clampi(i - 1, 0, cpar.nx)) * cpar.dx;
        const float y = static_cast<float>(clampi(j - 1, 0, cpar.ny)) * cpar.dy;
        const float z = static_cast<float>(clampi(k - 1, 0, cpar.nz)) * cpar.dz;
        if ((x >= 0.0f && x <= cpar.Lx / 3.0f) &&
            (y >= 0.0f && y <= cpar.Ly / 2.0f) &&
            (z >= 0.0f && z <= cpar.Lz / 3.0f)) {
            F[idx3d(i, j, k, nxTot, nyTot)] = 1.0f;
        }
    }
}

__global__ void cal_nu_rho_kernel(const float* __restrict__ F,
                                  float* __restrict__ rho,
                                  float* __restrict__ nu) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cpar.nTot) return;

    const float f = clamp01f(F[tid]);
    rho[tid] = cpar.rho_g * (1.0f - f) + cpar.rho_l * f;
    nu[tid] = cpar.nu_l * f + cpar.nu_g * (1.0f - f);
}

__global__ void set_bc_j_kernel(float* __restrict__ u,
                                float* __restrict__ v,
                                float* __restrict__ w,
                                float* __restrict__ F,
                                float* __restrict__ p,
                                float* __restrict__ rho) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;

    if (i >= nxTot || k >= nzTot) return;

    const int jb = cpar.jmin - 1;
    const int jt = cpar.jmax + 1;
    const int j0 = cpar.jmin;
    const int j1 = cpar.jmax;

    const size_t id_b_ghost = idx3d(i, jb, k, nxTot, nyTot);
    const size_t id_b_in    = idx3d(i, j0, k, nxTot, nyTot);
    const size_t id_t_ghost = idx3d(i, jt, k, nxTot, nyTot);
    const size_t id_t_in    = idx3d(i, j1, k, nxTot, nyTot);

    u[id_b_ghost] = u[id_b_in];
    v[id_b_in]    = 0.0f;
    w[id_b_ghost] = w[id_b_in];
    F[id_b_ghost] = F[id_b_in];
    p[id_b_ghost] = p[id_b_in];
    rho[id_b_ghost] = rho[id_b_in];

    u[id_t_ghost] = u[id_t_in];
    v[id_t_ghost] = 0.0f;
    w[id_t_ghost] = w[id_t_in];
    F[id_t_ghost] = F[id_t_in];
    p[id_t_ghost] = p[id_t_in];
    rho[id_t_ghost] = rho[id_t_in];
}

__global__ void set_bc_i_kernel(float* __restrict__ u,
                                float* __restrict__ v,
                                float* __restrict__ w,
                                float* __restrict__ F,
                                float* __restrict__ p,
                                float* __restrict__ rho) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;

    if (j >= nyTot || k >= nzTot) return;

    const int il = cpar.imin - 1;
    const int ir = cpar.imax + 1;
    const int i0 = cpar.imin;
    const int i1 = cpar.imax;

    const size_t id_l_ghost = idx3d(il, j, k, nxTot, nyTot);
    const size_t id_l_in    = idx3d(i0, j, k, nxTot, nyTot);
    const size_t id_r_ghost = idx3d(ir, j, k, nxTot, nyTot);
    const size_t id_r_in    = idx3d(i1, j, k, nxTot, nyTot);
    const size_t id_u_left  = idx3d(cpar.imin, j, k, nxTot, nyTot);
    const size_t id_u_right = idx3d(cpar.imax + 1, j, k, nxTot, nyTot);

    u[id_u_left] = 0.0f;
    v[id_l_ghost] = v[id_l_in];
    w[id_l_ghost] = w[id_l_in];
    F[id_l_ghost] = F[id_l_in];
    p[id_l_ghost] = p[id_l_in];
    rho[id_l_ghost] = rho[id_l_in];

    u[id_u_right] = 0.0f;
    v[id_r_ghost] = v[id_r_in];
    w[id_r_ghost] = w[id_r_in];
    F[id_r_ghost] = F[id_r_in];
    p[id_r_ghost] = p[id_r_in];
    rho[id_r_ghost] = rho[id_r_in];
}

__global__ void set_bc_k_kernel(float* __restrict__ u,
                                float* __restrict__ v,
                                float* __restrict__ w,
                                float* __restrict__ F,
                                float* __restrict__ p,
                                float* __restrict__ rho) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;

    if (i >= nxTot || j >= nyTot) return;

    const int kb = cpar.kmin - 1;
    const int kf = cpar.kmax + 1;
    const int k0 = cpar.kmin;
    const int k1 = cpar.kmax;

    const size_t id_b_ghost = idx3d(i, j, kb, nxTot, nyTot);
    const size_t id_b_in    = idx3d(i, j, k0, nxTot, nyTot);
    const size_t id_f_ghost = idx3d(i, j, kf, nxTot, nyTot);
    const size_t id_f_in    = idx3d(i, j, k1, nxTot, nyTot);
    const size_t id_w_front = idx3d(i, j, cpar.kmin, nxTot, nyTot);
    const size_t id_w_back  = idx3d(i, j, cpar.kmax + 1, nxTot, nyTot);

    u[id_b_ghost] = u[id_b_in];
    v[id_b_ghost] = v[id_b_in];
    w[id_w_front] = 0.0f;
    F[id_b_ghost] = F[id_b_in];
    p[id_b_ghost] = p[id_b_in];
    rho[id_b_ghost] = rho[id_b_in];

    u[id_f_ghost] = u[id_f_in];
    v[id_f_ghost] = v[id_f_in];
    w[id_w_back]  = 0.0f;
    F[id_f_ghost] = F[id_f_in];
    p[id_f_ghost] = p[id_f_in];
    rho[id_f_ghost] = rho[id_f_in];
}

__global__ void advect_u_kernel(const float* __restrict__ u,
                                const float* __restrict__ v,
                                const float* __restrict__ w,
                                const float* __restrict__ rho,
                                const float* __restrict__ nu,
                                float* __restrict__ u_star) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + (cpar.imin + 1);
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id   = idx3d(i, j, k, nxTot, nyTot);
    const size_t idim = idx3d(i - 1, j, k, nxTot, nyTot);
    const size_t idip = idx3d(i + 1, j, k, nxTot, nyTot);
    const size_t idjm = idx3d(i, j - 1, k, nxTot, nyTot);
    const size_t idjp = idx3d(i, j + 1, k, nxTot, nyTot);
    const size_t idkm = idx3d(i, j, k - 1, nxTot, nyTot);
    const size_t idkp = idx3d(i, j, k + 1, nxTot, nyTot);

    const float v_here = 0.25f * (
        v[idx3d(i - 1, j,     k, nxTot, nyTot)] +
        v[idx3d(i - 1, j + 1, k, nxTot, nyTot)] +
        v[idx3d(i,     j,     k, nxTot, nyTot)] +
        v[idx3d(i,     j + 1, k, nxTot, nyTot)]
    );
    const float w_here = 0.25f * (
        w[idx3d(i - 1, j, k,     nxTot, nyTot)] +
        w[idx3d(i - 1, j, k + 1, nxTot, nyTot)] +
        w[idx3d(i,     j, k,     nxTot, nyTot)] +
        w[idx3d(i,     j, k + 1, nxTot, nyTot)]
    );

    const float u_here = u[id];
    const float dudx = (u_here > 0.0f)
        ? (u_here - u[idim]) * cpar.dxi
        : (u[idip] - u_here) * cpar.dxi;
    const float dudy = (v_here > 0.0f)
        ? (u_here - u[idjm]) * cpar.dyi
        : (u[idjp] - u_here) * cpar.dyi;
    const float dudz = (w_here > 0.0f)
        ? (u_here - u[idkm]) * cpar.dzi
        : (u[idkp] - u_here) * cpar.dzi;

    const float visc =
        nu[id] * (u[idim] - 2.0f * u_here + u[idip]) * cpar.dxi * cpar.dxi +
        nu[id] * (u[idjm] - 2.0f * u_here + u[idjp]) * cpar.dyi * cpar.dyi +
        nu[id] * (u[idkm] - 2.0f * u_here + u[idkp]) * cpar.dzi * cpar.dzi;

    const float adv = -u_here * dudx - v_here * dudy - w_here * dudz;

    u_star[id] = u_here + cpar.dt * (visc + adv + cpar.gx);
}

__global__ void advect_v_kernel(const float* __restrict__ u,
                                const float* __restrict__ v,
                                const float* __restrict__ w,
                                const float* __restrict__ rho,
                                const float* __restrict__ nu,
                                float* __restrict__ v_star) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + (cpar.jmin + 1);
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id   = idx3d(i, j, k, nxTot, nyTot);
    const size_t idim = idx3d(i - 1, j, k, nxTot, nyTot);
    const size_t idip = idx3d(i + 1, j, k, nxTot, nyTot);
    const size_t idjm = idx3d(i, j - 1, k, nxTot, nyTot);
    const size_t idjp = idx3d(i, j + 1, k, nxTot, nyTot);
    const size_t idkm = idx3d(i, j, k - 1, nxTot, nyTot);
    const size_t idkp = idx3d(i, j, k + 1, nxTot, nyTot);

    const float u_here = 0.25f * (
        u[idx3d(i,     j - 1, k, nxTot, nyTot)] +
        u[idx3d(i,     j,     k, nxTot, nyTot)] +
        u[idx3d(i + 1, j - 1, k, nxTot, nyTot)] +
        u[idx3d(i + 1, j,     k, nxTot, nyTot)]
    );
    const float w_here = 0.25f * (
        w[idx3d(i, j - 1, k + 1, nxTot, nyTot)] +
        w[idx3d(i, j - 1, k,     nxTot, nyTot)] +
        w[idx3d(i, j,     k,     nxTot, nyTot)] +
        w[idx3d(i, j,     k + 1, nxTot, nyTot)]
    );

    const float v_here = v[id];
    const float dvdx = (u_here > 0.0f)
        ? (v_here - v[idim]) * cpar.dxi
        : (v[idip] - v_here) * cpar.dxi;
    const float dvdy = (v_here > 0.0f)
        ? (v_here - v[idjm]) * cpar.dyi
        : (v[idjp] - v_here) * cpar.dyi;
    const float dvdz = (w_here > 0.0f)
        ? (v_here - v[idkm]) * cpar.dzi
        : (v[idkp] - v_here) * cpar.dzi;

    const float visc =
        nu[id] * (v[idim] - 2.0f * v_here + v[idip]) * cpar.dxi * cpar.dxi +
        nu[id] * (v[idjm] - 2.0f * v_here + v[idjp]) * cpar.dyi * cpar.dyi +
        nu[id] * (v[idkm] - 2.0f * v_here + v[idkp]) * cpar.dzi * cpar.dzi;

    const float adv = -u_here * dvdx - v_here * dvdy - w_here * dvdz;

    v_star[id] = v_here + cpar.dt * (visc + adv + cpar.gy);
}

__global__ void advect_w_kernel(const float* __restrict__ u,
                                const float* __restrict__ v,
                                const float* __restrict__ w,
                                const float* __restrict__ rho,
                                const float* __restrict__ nu,
                                float* __restrict__ w_star) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + (cpar.kmin + 1);

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id   = idx3d(i, j, k, nxTot, nyTot);
    const size_t idim = idx3d(i - 1, j, k, nxTot, nyTot);
    const size_t idip = idx3d(i + 1, j, k, nxTot, nyTot);
    const size_t idjm = idx3d(i, j - 1, k, nxTot, nyTot);
    const size_t idjp = idx3d(i, j + 1, k, nxTot, nyTot);
    const size_t idkm = idx3d(i, j, k - 1, nxTot, nyTot);
    const size_t idkp = idx3d(i, j, k + 1, nxTot, nyTot);

    const float u_here = 0.25f * (
        u[idx3d(i + 1, j, k - 1, nxTot, nyTot)] +
        u[idx3d(i,     j, k - 1, nxTot, nyTot)] +
        u[idx3d(i + 1, j, k,     nxTot, nyTot)] +
        u[idx3d(i,     j, k,     nxTot, nyTot)]
    );
    const float v_here = 0.25f * (
        v[idx3d(i, j + 1, k - 1, nxTot, nyTot)] +
        v[idx3d(i, j,     k - 1, nxTot, nyTot)] +
        v[idx3d(i, j,     k,     nxTot, nyTot)] +
        v[idx3d(i, j + 1, k,     nxTot, nyTot)]
    );

    const float w_here = w[id];
    const float dwdx = (u_here > 0.0f)
        ? (w_here - w[idim]) * cpar.dxi
        : (w[idip] - w_here) * cpar.dxi;
    const float dwdy = (v_here > 0.0f)
        ? (w_here - w[idjm]) * cpar.dyi
        : (w[idjp] - w_here) * cpar.dyi;
    const float dwdz = (w_here > 0.0f)
        ? (w_here - w[idkm]) * cpar.dzi
        : (w[idkp] - w_here) * cpar.dzi;

    const float visc =
        nu[id] * (w[idim] - 2.0f * w_here + w[idip]) * cpar.dxi * cpar.dxi +
        nu[id] * (w[idjm] - 2.0f * w_here + w[idjp]) * cpar.dyi * cpar.dyi +
        nu[id] * (w[idkm] - 2.0f * w_here + w[idkp]) * cpar.dzi * cpar.dzi;

    const float adv = -u_here * dwdx - v_here * dwdy - w_here * dwdz;

    w_star[id] = w_here + cpar.dt * (visc + adv + cpar.gz);
}

__global__ void solve_p_jacobi_kernel(const float* __restrict__ u_star,
                                      const float* __restrict__ v_star,
                                      const float* __restrict__ w_star,
                                      const float* __restrict__ rho,
                                      const float* __restrict__ p,
                                      float* __restrict__ pt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float rhs =
        rho[id] / cpar.dt *
        ((u_star[idx3d(i + 1, j,     k,     nxTot, nyTot)] - u_star[id]) * cpar.dxi +
         (v_star[idx3d(i,     j + 1, k,     nxTot, nyTot)] - v_star[id]) * cpar.dyi +
         (w_star[idx3d(i,     j,     k + 1, nxTot, nyTot)] - w_star[id]) * cpar.dzi);

    const float ae = (i != cpar.imax) ? (cpar.dxi * cpar.dxi) : 0.0f;
    const float aw = (i != cpar.imin) ? (cpar.dxi * cpar.dxi) : 0.0f;
    const float an = (j != cpar.jmax) ? (cpar.dyi * cpar.dyi) : 0.0f;
    const float as = (j != cpar.jmin) ? (cpar.dyi * cpar.dyi) : 0.0f;
    const float af = (k != cpar.kmax) ? (cpar.dzi * cpar.dzi) : 0.0f;
    const float ab = (k != cpar.kmin) ? (cpar.dzi * cpar.dzi) : 0.0f;
    const float ap = -(ae + aw + an + as + af + ab);

    pt[id] = (rhs
             - ae * p[idx3d(i + 1, j,     k,     nxTot, nyTot)]
             - aw * p[idx3d(i - 1, j,     k,     nxTot, nyTot)]
             - an * p[idx3d(i,     j + 1, k,     nxTot, nyTot)]
             - as * p[idx3d(i,     j - 1, k,     nxTot, nyTot)]
             - af * p[idx3d(i,     j,     k + 1, nxTot, nyTot)]
             - ab * p[idx3d(i,     j,     k - 1, nxTot, nyTot)]) / ap;
}

__global__ void update_u_kernel(const float* __restrict__ rho,
                                const float* __restrict__ u_star,
                                const float* __restrict__ p,
                                float* __restrict__ u) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + (cpar.imin + 1);
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idm = idx3d(i - 1, j, k, nxTot, nyTot);
    const float r = 0.5f * (rho[id] + rho[idm]);
    u[id] = u_star[id] - cpar.dt / r * (p[id] - p[idm]) * cpar.dxi;
}

__global__ void update_v_kernel(const float* __restrict__ rho,
                                const float* __restrict__ v_star,
                                const float* __restrict__ p,
                                float* __restrict__ v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + (cpar.jmin + 1);
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idm = idx3d(i, j - 1, k, nxTot, nyTot);
    const float r = 0.5f * (rho[id] + rho[idm]);
    v[id] = v_star[id] - cpar.dt / r * (p[id] - p[idm]) * cpar.dyi;
}

__global__ void update_w_kernel(const float* __restrict__ rho,
                                const float* __restrict__ w_star,
                                const float* __restrict__ p,
                                float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + (cpar.kmin + 1);

    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idm = idx3d(i, j, k - 1, nxTot, nyTot);
    const float r = 0.5f * (rho[id] + rho[idm]);
    w[id] = w_star[id] - cpar.dt / r * (p[id] - p[idm]) * cpar.dzi;
}

__global__ void copy_field_kernel(const float* __restrict__ src, float* __restrict__ dst) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cpar.nTot) {
        dst[tid] = src[tid];
    }
}

__global__ void post_process_f_kernel(float* __restrict__ F) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cpar.nTot) {
        F[tid] = clamp01f(F[tid]);
    }
}

__global__ void fct_x_sweep_stage1_kernel(const float* __restrict__ F,
                                          const float* __restrict__ u,
                                          float* __restrict__ Ftd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dy * cpar.dz *
        (u[idx3d(i + 1, j, k, nxTot, nyTot)] - u[idx3d(i, j, k, nxTot, nyTot)]);

    const float ui  = u[idx3d(i,     j, k, nxTot, nyTot)];
    const float uip = u[idx3d(i + 1, j, k, nxTot, nyTot)];

    const float fl_L = (ui  >= 0.0f) ? ui  * cpar.dt * F[idx3d(i - 1, j, k, nxTot, nyTot)]
                                     : ui  * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)];
    const float fr_L = (uip >= 0.0f) ? uip * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)]
                                     : uip * cpar.dt * F[idx3d(i + 1, j, k, nxTot, nyTot)];

    float out = (F[id] + (fl_L - fr_L) * cpar.dy * cpar.dz / vol) * vol / dv;
    if (out > 1.0f || out < 0.0f) out = clamp01f(out);
    Ftd[id] = out;
}

__global__ void fct_x_sweep_stage2_kernel(const float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ u,
                                          float* __restrict__ ax,
                                          float* __restrict__ ay,
                                          float* __restrict__ rp,
                                          float* __restrict__ rm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float fmax = fmaxf(Ftd[id], fmaxf(Ftd[idx3d(i - 1, j, k, nxTot, nyTot)],
                                            Ftd[idx3d(i + 1, j, k, nxTot, nyTot)]));
    const float fmin = fminf(Ftd[id], fminf(Ftd[idx3d(i - 1, j, k, nxTot, nyTot)],
                                            Ftd[idx3d(i + 1, j, k, nxTot, nyTot)]));

    const float ui  = u[idx3d(i,     j, k, nxTot, nyTot)];
    const float uip = u[idx3d(i + 1, j, k, nxTot, nyTot)];

    const float fl_L = (ui  >= 0.0f) ? ui  * cpar.dt * F[idx3d(i - 1, j, k, nxTot, nyTot)]
                                     : ui  * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)];
    const float fr_L = (uip >= 0.0f) ? uip * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)]
                                     : uip * cpar.dt * F[idx3d(i + 1, j, k, nxTot, nyTot)];

    const float fl_H = (ui  <= 0.0f) ? ui  * cpar.dt * F[idx3d(i - 1, j, k, nxTot, nyTot)]
                                     : ui  * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)];
    const float fr_H = (uip <= 0.0f) ? uip * cpar.dt * F[idx3d(i,     j, k, nxTot, nyTot)]
                                     : uip * cpar.dt * F[idx3d(i + 1, j, k, nxTot, nyTot)];

    ax[idx3d(i + 1, j, k, nxTot, nyTot)] = fr_H - fr_L;
    ax[id] = fl_H - fl_L;
    ay[idx3d(i, j + 1, k, nxTot, nyTot)] = 0.0f;
    ay[id] = 0.0f;

    const float pp = fmaxf(0.0f, ax[id]) - fminf(0.0f, ax[idx3d(i + 1, j, k, nxTot, nyTot)]) +
                     fmaxf(0.0f, ay[id]) - fminf(0.0f, ay[idx3d(i, j + 1, k, nxTot, nyTot)]);
    const float qp = (fmax - Ftd[id]) * cpar.dx;
    rp[id] = (pp > 0.0f) ? fminf(1.0f, qp / pp) : 0.0f;

    const float pm = fmaxf(0.0f, ax[idx3d(i + 1, j, k, nxTot, nyTot)]) - fminf(0.0f, ax[id]) +
                     fmaxf(0.0f, ay[idx3d(i, j + 1, k, nxTot, nyTot)]) - fminf(0.0f, ay[id]);
    const float qm = (Ftd[id] - fmin) * cpar.dx;
    rm[id] = (pm > 0.0f) ? fminf(1.0f, qm / pm) : 0.0f;
}

__global__ void fct_x_sweep_stage3_kernel(const float* __restrict__ ax,
                                          const float* __restrict__ ay,
                                          const float* __restrict__ rp,
                                          const float* __restrict__ rm,
                                          float* __restrict__ cx,
                                          float* __restrict__ cy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idxp = idx3d(i + 1, j, k, nxTot, nyTot);
    const size_t idyp = idx3d(i, j + 1, k, nxTot, nyTot);

    if (ax[idxp] >= 0.0f) {
        cx[idxp] = fminf(rp[idxp], rm[id]);
    } else {
        cx[idxp] = fminf(rp[id], rm[idxp]);
    }

    if (ay[idyp] >= 0.0f) {
        cy[idyp] = fminf(rp[idyp], rm[id]);
    } else {
        cy[idyp] = fminf(rp[id], rm[idyp]);
    }
}

__global__ void fct_x_sweep_stage4_kernel(float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ u,
                                          const float* __restrict__ ax,
                                          const float* __restrict__ ay,
                                          const float* __restrict__ cx,
                                          const float* __restrict__ cy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dy * cpar.dz *
        (u[idx3d(i + 1, j, k, nxTot, nyTot)] - u[id]);

    float out = Ftd[id] - (
        (ax[idx3d(i + 1, j, k, nxTot, nyTot)] * cx[idx3d(i + 1, j, k, nxTot, nyTot)] -
         ax[id] * cx[id] +
         ay[idx3d(i, j + 1, k, nxTot, nyTot)] * cy[idx3d(i, j + 1, k, nxTot, nyTot)] -
         ay[id] * cy[id]) / cpar.dy) * vol / dv;

    F[id] = clamp01f(out);
}

__global__ void fct_y_sweep_stage1_kernel(const float* __restrict__ F,
                                          const float* __restrict__ v,
                                          float* __restrict__ Ftd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dx * cpar.dz *
        (v[idx3d(i, j + 1, k, nxTot, nyTot)] - v[id]);

    const float vj  = v[id];
    const float vjp = v[idx3d(i, j + 1, k, nxTot, nyTot)];

    const float ft_L = (vjp >= 0.0f) ? vjp * cpar.dt * F[id]
                                     : vjp * cpar.dt * F[idx3d(i, j + 1, k, nxTot, nyTot)];
    const float fb_L = (vj  >= 0.0f) ? vj  * cpar.dt * F[idx3d(i, j - 1, k, nxTot, nyTot)]
                                     : vj  * cpar.dt * F[id];

    float out = (F[id] + (fb_L - ft_L) * cpar.dy / (cpar.dx * cpar.dy)) * vol / dv;
    if (out > 1.0f || out < 0.0f) out = clamp01f(out);
    Ftd[id] = out;
}

__global__ void fct_y_sweep_stage2_kernel(const float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ v,
                                          float* __restrict__ ax,
                                          float* __restrict__ ay,
                                          float* __restrict__ rp,
                                          float* __restrict__ rm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float fmax = fmaxf(Ftd[id], fmaxf(Ftd[idx3d(i, j - 1, k, nxTot, nyTot)],
                                            Ftd[idx3d(i, j + 1, k, nxTot, nyTot)]));
    const float fmin = fminf(Ftd[id], fminf(Ftd[idx3d(i, j - 1, k, nxTot, nyTot)],
                                            Ftd[idx3d(i, j + 1, k, nxTot, nyTot)]));

    const float vj  = v[id];
    const float vjp = v[idx3d(i, j + 1, k, nxTot, nyTot)];

    const float ft_L = (vjp >= 0.0f) ? vjp * cpar.dt * F[id]
                                     : vjp * cpar.dt * F[idx3d(i, j + 1, k, nxTot, nyTot)];
    const float fb_L = (vj  >= 0.0f) ? vj  * cpar.dt * F[idx3d(i, j - 1, k, nxTot, nyTot)]
                                     : vj  * cpar.dt * F[id];

    const float ft_H = (vjp <= 0.0f) ? vjp * cpar.dt * F[id]
                                     : vjp * cpar.dt * F[idx3d(i, j + 1, k, nxTot, nyTot)];
    const float fb_H = (vj  <= 0.0f) ? vj  * cpar.dt * F[idx3d(i, j - 1, k, nxTot, nyTot)]
                                     : vj  * cpar.dt * F[id];

    ax[idx3d(i + 1, j, k, nxTot, nyTot)] = 0.0f;
    ax[id] = 0.0f;
    ay[idx3d(i, j + 1, k, nxTot, nyTot)] = ft_H - ft_L;
    ay[id] = fb_H - fb_L;

    const float pp = fmaxf(0.0f, ax[id]) - fminf(0.0f, ax[idx3d(i + 1, j, k, nxTot, nyTot)]) +
                     fmaxf(0.0f, ay[id]) - fminf(0.0f, ay[idx3d(i, j + 1, k, nxTot, nyTot)]);
    const float qp = (fmax - Ftd[id]) * cpar.dx;
    rp[id] = (pp > 0.0f) ? fminf(1.0f, qp / pp) : 0.0f;

    const float pm = fmaxf(0.0f, ax[idx3d(i + 1, j, k, nxTot, nyTot)]) - fminf(0.0f, ax[id]) +
                     fmaxf(0.0f, ay[idx3d(i, j + 1, k, nxTot, nyTot)]) - fminf(0.0f, ay[id]);
    const float qm = (Ftd[id] - fmin) * cpar.dx;
    rm[id] = (pm > 0.0f) ? fminf(1.0f, qm / pm) : 0.0f;
}

__global__ void fct_y_sweep_stage3_kernel(const float* __restrict__ ax,
                                          const float* __restrict__ ay,
                                          const float* __restrict__ rp,
                                          const float* __restrict__ rm,
                                          float* __restrict__ cx,
                                          float* __restrict__ cy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;

    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idxp = idx3d(i + 1, j, k, nxTot, nyTot);
    const size_t idyp = idx3d(i, j + 1, k, nxTot, nyTot);

    if (ax[idxp] >= 0.0f) {
        cx[idxp] = fminf(rp[idxp], rm[id]);
    } else {
        cx[idxp] = fminf(rp[id], rm[idxp]);
    }

    if (ay[idyp] >= 0.0f) {
        cy[idyp] = fminf(rp[idyp], rm[id]);
    } else {
        cy[idyp] = fminf(rp[id], rm[idyp]);
    }
}

__global__ void fct_y_sweep_stage4_kernel(float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ v,
                                          const float* __restrict__ ax,
                                          const float* __restrict__ ay,
                                          const float* __restrict__ cx,
                                          const float* __restrict__ cy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dx * cpar.dz *
        (v[idx3d(i, j + 1, k, nxTot, nyTot)] - v[id]);

    float out = Ftd[id] - (
        (ax[idx3d(i + 1, j, k, nxTot, nyTot)] * cx[idx3d(i + 1, j, k, nxTot, nyTot)] -
         ax[id] * cx[id] +
         ay[idx3d(i, j + 1, k, nxTot, nyTot)] * cy[idx3d(i, j + 1, k, nxTot, nyTot)] -
         ay[id] * cy[id]) / cpar.dy) * vol / dv;

    F[id] = clamp01f(out);
}

__global__ void fct_z_sweep_stage1_kernel(const float* __restrict__ F,
                                          const float* __restrict__ w,
                                          float* __restrict__ Ftd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dx * cpar.dy *
        (w[idx3d(i, j, k + 1, nxTot, nyTot)] - w[id]);

    const float wk  = w[id];
    const float wkp = w[idx3d(i, j, k + 1, nxTot, nyTot)];

    const float ff_L = (wkp >= 0.0f) ? wkp * cpar.dt * F[id]
                                     : wkp * cpar.dt * F[idx3d(i, j, k + 1, nxTot, nyTot)];
    const float fb_L = (wk  >= 0.0f) ? wk  * cpar.dt * F[idx3d(i, j, k - 1, nxTot, nyTot)]
                                     : wk  * cpar.dt * F[id];

    float out = (F[id] + (fb_L - ff_L) * cpar.dy * cpar.dx / vol) * vol / dv;
    if (out > 1.0f || out < 0.0f) out = clamp01f(out);
    Ftd[id] = out;
}

__global__ void fct_z_sweep_stage2_kernel(const float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ w,
                                          float* __restrict__ az,
                                          float* __restrict__ rp,
                                          float* __restrict__ rm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float fmax = fmaxf(Ftd[id], fmaxf(Ftd[idx3d(i, j, k - 1, nxTot, nyTot)],
                                            Ftd[idx3d(i, j, k + 1, nxTot, nyTot)]));
    const float fmin = fminf(Ftd[id], fminf(Ftd[idx3d(i, j, k - 1, nxTot, nyTot)],
                                            Ftd[idx3d(i, j, k + 1, nxTot, nyTot)]));

    const float wk  = w[id];
    const float wkp = w[idx3d(i, j, k + 1, nxTot, nyTot)];

    const float ff_L = (wkp >= 0.0f) ? wkp * cpar.dt * F[id]
                                     : wkp * cpar.dt * F[idx3d(i, j, k + 1, nxTot, nyTot)];
    const float fb_L = (wk  >= 0.0f) ? wk  * cpar.dt * F[idx3d(i, j, k - 1, nxTot, nyTot)]
                                     : wk  * cpar.dt * F[id];

    const float ff_H = (wkp <= 0.0f) ? wkp * cpar.dt * F[id]
                                     : wkp * cpar.dt * F[idx3d(i, j, k + 1, nxTot, nyTot)];
    const float fb_H = (wk  <= 0.0f) ? wk  * cpar.dt * F[idx3d(i, j, k - 1, nxTot, nyTot)]
                                     : wk  * cpar.dt * F[id];

    az[idx3d(i, j, k + 1, nxTot, nyTot)] = ff_H - ff_L;
    az[id] = fb_H - fb_L;

    const float pp = fmaxf(0.0f, az[id]) - fminf(0.0f, az[idx3d(i, j, k + 1, nxTot, nyTot)]);
    const float qp = (fmax - Ftd[id]) * cpar.dz;
    rp[id] = (pp > 0.0f) ? fminf(1.0f, qp / pp) : 0.0f;

    const float pm = fmaxf(0.0f, az[idx3d(i, j, k + 1, nxTot, nyTot)]) - fminf(0.0f, az[id]);
    const float qm = (Ftd[id] - fmin) * cpar.dz;
    rm[id] = (pm > 0.0f) ? fminf(1.0f, qm / pm) : 0.0f;
}

__global__ void fct_z_sweep_stage3_kernel(const float* __restrict__ az,
                                          const float* __restrict__ rp,
                                          const float* __restrict__ rm,
                                          float* __restrict__ cz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);
    const size_t idzp = idx3d(i, j, k + 1, nxTot, nyTot);

    if (az[idzp] >= 0.0f) {
        cz[idzp] = fminf(rp[idzp], rm[id]);
    } else {
        cz[idzp] = fminf(rp[id], rm[idzp]);
    }
}

__global__ void fct_z_sweep_stage4_kernel(float* __restrict__ F,
                                          const float* __restrict__ Ftd,
                                          const float* __restrict__ w,
                                          const float* __restrict__ az,
                                          const float* __restrict__ cz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + cpar.imin;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + cpar.jmin;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + cpar.kmin;
    if (i > cpar.imax || j > cpar.jmax || k > cpar.kmax) return;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const size_t id = idx3d(i, j, k, nxTot, nyTot);

    const float vol = cpar.dx * cpar.dy * cpar.dz;
    const float dv = vol - cpar.dt * cpar.dx * cpar.dy *
        (w[idx3d(i, j, k + 1, nxTot, nyTot)] - w[id]);

    float out = Ftd[id] - (
        (az[idx3d(i, j, k + 1, nxTot, nyTot)] * cz[idx3d(i, j, k + 1, nxTot, nyTot)] -
         az[id] * cz[id]) / cpar.dz) * vol / dv;

    F[id] = clamp01f(out);
}

void write_legacy_vtk(const std::string& filename,
                      const std::vector<float>& field,
                      const int nxTot,
                      const int nyTot,
                      const int nzTot) {
    std::ofstream out(filename, std::ios::out);
    if (!out) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const float sx = 1.0f / static_cast<float>(nxTot - 1);
    const float sy = 1.0f / static_cast<float>(nyTot - 1);
    const float sz = 1.0f / static_cast<float>(nzTot - 1);

    out << "# vtk DataFile Version 3.0\n";
    out << "3D VOF field\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nxTot << " " << nyTot << " " << nzTot << "\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING " << sx << " " << sy << " " << sz << "\n";
    out << "POINT_DATA " << static_cast<long long>(nxTot) * nyTot * nzTot << "\n";
    out << "SCALARS VOF float 1\n";
    out << "LOOKUP_TABLE default\n";

    out << std::setprecision(8);
    for (int k = 0; k < nzTot; ++k) {
        for (int j = 0; j < nyTot; ++j) {
            for (int i = 0; i < nxTot; ++i) {
                out << field[idx3d(i, j, k, nxTot, nyTot)] << "\n";
            }
        }
    }
}

void apply_bc(float* u, float* v, float* w, float* F, float* p, float* rho) {
    const dim3 block2d(16, 16);
    const dim3 grid_j((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_i((cpar_host.ny + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_k((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.ny + 2 + block2d.y - 1) / block2d.y);

    set_bc_j_kernel<<<grid_j, block2d>>>(u, v, w, F, p, rho);
    set_bc_i_kernel<<<grid_i, block2d>>>(u, v, w, F, p, rho);
    set_bc_k_kernel<<<grid_k, block2d>>>(u, v, w, F, p, rho);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void set_bc_vel_j_kernel(float* __restrict__ u,
                                    float* __restrict__ v,
                                    float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (i >= nxTot || k >= nzTot) return;

    const int jb = cpar.jmin - 1;
    const int jt = cpar.jmax + 1;
    const int j0 = cpar.jmin;
    const int j1 = cpar.jmax;

    const size_t id_b_ghost = idx3d(i, jb, k, nxTot, nyTot);
    const size_t id_b_in    = idx3d(i, j0, k, nxTot, nyTot);
    const size_t id_t_ghost = idx3d(i, jt, k, nxTot, nyTot);
    const size_t id_t_in    = idx3d(i, j1, k, nxTot, nyTot);

    u[id_b_ghost] = u[id_b_in];
    v[id_b_in]    = 0.0f;
    w[id_b_ghost] = w[id_b_in];

    u[id_t_ghost] = u[id_t_in];
    v[id_t_ghost] = 0.0f;
    w[id_t_ghost] = w[id_t_in];
}

__global__ void set_bc_vel_i_kernel(float* __restrict__ u,
                                    float* __restrict__ v,
                                    float* __restrict__ w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (j >= nyTot || k >= nzTot) return;

    const int il = cpar.imin - 1;
    const int ir = cpar.imax + 1;
    const int i0 = cpar.imin;
    const int i1 = cpar.imax;

    const size_t id_l_ghost = idx3d(il, j, k, nxTot, nyTot);
    const size_t id_l_in    = idx3d(i0, j, k, nxTot, nyTot);
    const size_t id_r_ghost = idx3d(ir, j, k, nxTot, nyTot);
    const size_t id_r_in    = idx3d(i1, j, k, nxTot, nyTot);
    const size_t id_u_left  = idx3d(cpar.imin, j, k, nxTot, nyTot);
    const size_t id_u_right = idx3d(cpar.imax + 1, j, k, nxTot, nyTot);

    u[id_u_left] = 0.0f;
    v[id_l_ghost] = v[id_l_in];
    w[id_l_ghost] = w[id_l_in];

    u[id_u_right] = 0.0f;
    v[id_r_ghost] = v[id_r_in];
    w[id_r_ghost] = w[id_r_in];
}

__global__ void set_bc_vel_k_kernel(float* __restrict__ u,
                                    float* __restrict__ v,
                                    float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (i >= nxTot || j >= nyTot) return;

    const int kb = cpar.kmin - 1;
    const int kf = cpar.kmax + 1;
    const int k0 = cpar.kmin;
    const int k1 = cpar.kmax;

    const size_t id_b_ghost = idx3d(i, j, kb, nxTot, nyTot);
    const size_t id_b_in    = idx3d(i, j, k0, nxTot, nyTot);
    const size_t id_f_ghost = idx3d(i, j, kf, nxTot, nyTot);
    const size_t id_f_in    = idx3d(i, j, k1, nxTot, nyTot);
    const size_t id_w_front = idx3d(i, j, cpar.kmin, nxTot, nyTot);
    const size_t id_w_back  = idx3d(i, j, cpar.kmax + 1, nxTot, nyTot);

    u[id_b_ghost] = u[id_b_in];
    v[id_b_ghost] = v[id_b_in];
    w[id_w_front] = 0.0f;

    u[id_f_ghost] = u[id_f_in];
    v[id_f_ghost] = v[id_f_in];
    w[id_w_back]  = 0.0f;
}

__global__ void set_bc_F_j_kernel(float* __restrict__ F) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (i >= nxTot || k >= nzTot) return;

    const int jb = cpar.jmin - 1;
    const int jt = cpar.jmax + 1;
    const int j0 = cpar.jmin;
    const int j1 = cpar.jmax;

    F[idx3d(i, jb, k, nxTot, nyTot)] = F[idx3d(i, j0, k, nxTot, nyTot)];
    F[idx3d(i, jt, k, nxTot, nyTot)] = F[idx3d(i, j1, k, nxTot, nyTot)];
}

__global__ void set_bc_F_i_kernel(float* __restrict__ F) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (j >= nyTot || k >= nzTot) return;

    const int il = cpar.imin - 1;
    const int ir = cpar.imax + 1;
    const int i0 = cpar.imin;
    const int i1 = cpar.imax;

    F[idx3d(il, j, k, nxTot, nyTot)] = F[idx3d(i0, j, k, nxTot, nyTot)];
    F[idx3d(ir, j, k, nxTot, nyTot)] = F[idx3d(i1, j, k, nxTot, nyTot)];
}

__global__ void set_bc_F_k_kernel(float* __restrict__ F) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int nxTot = cpar.nx + 2;
    const int nyTot = cpar.ny + 2;
    const int nzTot = cpar.nz + 2;
    if (i >= nxTot || j >= nyTot) return;

    const int kb = cpar.kmin - 1;
    const int kf = cpar.kmax + 1;
    const int k0 = cpar.kmin;
    const int k1 = cpar.kmax;

    F[idx3d(i, j, kb, nxTot, nyTot)] = F[idx3d(i, j, k0, nxTot, nyTot)];
    F[idx3d(i, j, kf, nxTot, nyTot)] = F[idx3d(i, j, k1, nxTot, nyTot)];
}

void apply_bc_velocity(float* u, float* v, float* w) {
    const dim3 block2d(16, 16);
    const dim3 grid_j((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_i((cpar_host.ny + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_k((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.ny + 2 + block2d.y - 1) / block2d.y);

    set_bc_vel_j_kernel<<<grid_j, block2d>>>(u, v, w);
    set_bc_vel_i_kernel<<<grid_i, block2d>>>(u, v, w);
    set_bc_vel_k_kernel<<<grid_k, block2d>>>(u, v, w);
    CHECK_CUDA(cudaPeekAtLastError());
}

void apply_bc_F(float* F) {
    const dim3 block2d(16, 16);
    const dim3 grid_j((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_i((cpar_host.ny + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.nz + 2 + block2d.y - 1) / block2d.y);
    const dim3 grid_k((cpar_host.nx + 2 + block2d.x - 1) / block2d.x,
                      (cpar_host.ny + 2 + block2d.y - 1) / block2d.y);

    set_bc_F_j_kernel<<<grid_j, block2d>>>(F);
    set_bc_F_i_kernel<<<grid_i, block2d>>>(F);
    set_bc_F_k_kernel<<<grid_k, block2d>>>(F);
    CHECK_CUDA(cudaPeekAtLastError());
}

void run_fct_x(float* F, float* Ftd, float* u, float* ax, float* ay, float* cx, float* cy, float* rp, float* rm) {
    const dim3 block3d(8, 8, 4);
    const dim3 grid3d((cpar_host.nx + block3d.x - 1) / block3d.x,
                      (cpar_host.ny + block3d.y - 1) / block3d.y,
                      (cpar_host.nz + block3d.z - 1) / block3d.z);

    fct_x_sweep_stage1_kernel<<<grid3d, block3d>>>(F, u, Ftd);
    fct_x_sweep_stage2_kernel<<<grid3d, block3d>>>(F, Ftd, u, ax, ay, rp, rm);
    fct_x_sweep_stage3_kernel<<<grid3d, block3d>>>(ax, ay, rp, rm, cx, cy);
    fct_x_sweep_stage4_kernel<<<grid3d, block3d>>>(F, Ftd, u, ax, ay, cx, cy);
    CHECK_CUDA(cudaPeekAtLastError());
}

void run_fct_y(float* F, float* Ftd, float* v, float* ax, float* ay, float* cx, float* cy, float* rp, float* rm) {
    const dim3 block3d(8, 8, 4);
    const dim3 grid3d((cpar_host.nx + block3d.x - 1) / block3d.x,
                      (cpar_host.ny + block3d.y - 1) / block3d.y,
                      (cpar_host.nz + block3d.z - 1) / block3d.z);

    fct_y_sweep_stage1_kernel<<<grid3d, block3d>>>(F, v, Ftd);
    fct_y_sweep_stage2_kernel<<<grid3d, block3d>>>(F, Ftd, v, ax, ay, rp, rm);
    fct_y_sweep_stage3_kernel<<<grid3d, block3d>>>(ax, ay, rp, rm, cx, cy);
    fct_y_sweep_stage4_kernel<<<grid3d, block3d>>>(F, Ftd, v, ax, ay, cx, cy);
    CHECK_CUDA(cudaPeekAtLastError());
}

void run_fct_z(float* F, float* Ftd, float* w, float* az, float* cz, float* rp, float* rm) {
    const dim3 block3d(8, 8, 4);
    const dim3 grid3d((cpar_host.nx + block3d.x - 1) / block3d.x,
                      (cpar_host.ny + block3d.y - 1) / block3d.y,
                      (cpar_host.nz + block3d.z - 1) / block3d.z);

    fct_z_sweep_stage1_kernel<<<grid3d, block3d>>>(F, w, Ftd);
    fct_z_sweep_stage2_kernel<<<grid3d, block3d>>>(F, Ftd, w, az, rp, rm);
    fct_z_sweep_stage3_kernel<<<grid3d, block3d>>>(az, rp, rm, cz);
    fct_z_sweep_stage4_kernel<<<grid3d, block3d>>>(F, Ftd, w, az, cz);
    CHECK_CUDA(cudaPeekAtLastError());
}

struct DeviceFields {
    float* F = nullptr;
    float* Ftd = nullptr;
    float* ax = nullptr;
    float* ay = nullptr;
    float* az = nullptr;
    float* cx = nullptr;
    float* cy = nullptr;
    float* cz = nullptr;
    float* rp = nullptr;
    float* rm = nullptr;

    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* u_star = nullptr;
    float* v_star = nullptr;
    float* w_star = nullptr;
    float* p = nullptr;
    float* pt = nullptr;
    float* rho = nullptr;
    float* nu = nullptr;
};

void alloc_field(float** ptr, const size_t bytes) {
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(ptr), bytes));
    CHECK_CUDA(cudaMemset(*ptr, 0, bytes));
}

void free_all(DeviceFields& d) {
    cudaFree(d.F);
    cudaFree(d.Ftd);
    cudaFree(d.ax);
    cudaFree(d.ay);
    cudaFree(d.az);
    cudaFree(d.cx);
    cudaFree(d.cy);
    cudaFree(d.cz);
    cudaFree(d.rp);
    cudaFree(d.rm);
    cudaFree(d.u);
    cudaFree(d.v);
    cudaFree(d.w);
    cudaFree(d.u_star);
    cudaFree(d.v_star);
    cudaFree(d.w_star);
    cudaFree(d.p);
    cudaFree(d.pt);
    cudaFree(d.rho);
    cudaFree(d.nu);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);

    int initial_condition = 1;
    int nstep = 100;
    int max_steps = 0;

    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "-ic" && a + 1 < argc) {
            initial_condition = std::atoi(argv[++a]);
        } else if (arg == "--nstep" && a + 1 < argc) {
            nstep = std::atoi(argv[++a]);
        } else if (arg == "--max_steps" && a + 1 < argc) {
            max_steps = std::atoi(argv[++a]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./3dvof_cuda [-ic 1|2|3] [--nstep N] [--max_steps N]\n"
                << "  -ic         Initial condition selector. Only 1 is implemented, exactly as in the source.\n"
                << "  --nstep     Dump interval.\n"
                << "  --max_steps Number of steps to run. 0 means run until Ctrl+C.\n";
            return 0;
        }
    }

    if (initial_condition < 1 || initial_condition > 3) {
        std::cerr << "Invalid initial condition. Allowed values: 1, 2, 3" << std::endl;
        return EXIT_FAILURE;
    }

    constexpr int nx = 200;
    constexpr int ny = 200;
    constexpr int nz = 200;

    cpar_host.nx = nx;
    cpar_host.ny = ny;
    cpar_host.nz = nz;
    cpar_host.imin = 1;
    cpar_host.imax = cpar_host.imin + nx - 1;
    cpar_host.jmin = 1;
    cpar_host.jmax = cpar_host.jmin + ny - 1;
    cpar_host.kmin = 1;
    cpar_host.kmax = cpar_host.kmin + nz - 1;

    cpar_host.Lx = 0.1f;
    cpar_host.Ly = 0.1f;
    cpar_host.Lz = 0.1f;
    cpar_host.rho_l = 1000.0f;
    cpar_host.rho_g = 50.0f;
    cpar_host.nu_l = 1.0e-6f;
    cpar_host.nu_g = 1.5e-5f;
    cpar_host.sigma = 0.007f;
    cpar_host.gx = 0.0f;
    cpar_host.gy = -5.0f;
    cpar_host.gz = 0.0f;
    cpar_host.dt = 4.0e-6f;

    cpar_host.dx = cpar_host.Lx / static_cast<float>(nx);
    cpar_host.dy = cpar_host.Ly / static_cast<float>(ny);
    cpar_host.dz = cpar_host.Lz / static_cast<float>(nz);
    cpar_host.dxi = 1.0f / cpar_host.dx;
    cpar_host.dyi = 1.0f / cpar_host.dy;
    cpar_host.dzi = 1.0f / cpar_host.dz;
    cpar_host.nTot = (nx + 2) * (ny + 2) * (nz + 2);

    CHECK_CUDA(cudaMemcpyToSymbol(cpar, &cpar_host, sizeof(Params)));

    std::cout << ">>> A 3D VOF solver ported to CUDA (speed-tuned path)\n";
    std::cout << ">>> Grid resolution: " << nx << " x " << ny << " x " << nz
              << ", dt = " << std::scientific << cpar_host.dt << std::defaultfloat << "\n";
    std::cout << ">>> Density ratio: " << (cpar_host.rho_l / cpar_host.rho_g)
              << ", gravity : " << cpar_host.gy
              << ", sigma : " << cpar_host.sigma << "\n";
    std::cout << ">>> Viscosity ratio: " << (cpar_host.nu_l / cpar_host.nu_g) << "\n";

    DeviceFields d;
    const size_t bytes = static_cast<size_t>(cpar_host.nTot) * sizeof(float);

    alloc_field(&d.F, bytes);
    alloc_field(&d.Ftd, bytes);
    alloc_field(&d.ax, bytes);
    alloc_field(&d.ay, bytes);
    alloc_field(&d.az, bytes);
    alloc_field(&d.cx, bytes);
    alloc_field(&d.cy, bytes);
    alloc_field(&d.cz, bytes);
    alloc_field(&d.rp, bytes);
    alloc_field(&d.rm, bytes);

    alloc_field(&d.u, bytes);
    alloc_field(&d.v, bytes);
    alloc_field(&d.w, bytes);
    alloc_field(&d.u_star, bytes);
    alloc_field(&d.v_star, bytes);
    alloc_field(&d.w_star, bytes);
    alloc_field(&d.p, bytes);
    alloc_field(&d.pt, bytes);
    alloc_field(&d.rho, bytes);
    alloc_field(&d.nu, bytes);

    const dim3 blockInit(8, 8, 4);
    const dim3 gridInit((nx + 2 + blockInit.x - 1) / blockInit.x,
                        (ny + 2 + blockInit.y - 1) / blockInit.y,
                        (nz + 2 + blockInit.z - 1) / blockInit.z);

    set_init_F_kernel<<<gridInit, blockInit>>>(d.F, initial_condition);
    CHECK_CUDA(cudaPeekAtLastError());
    apply_bc_F(d.F);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::system("mkdir -p output");

    std::vector<float> F_host(static_cast<size_t>(cpar_host.nTot), 0.0f);

    const dim3 block3d(8, 8, 4);
    const dim3 gridXYZ((nx + block3d.x - 1) / block3d.x,
                       (ny + block3d.y - 1) / block3d.y,
                       (nz + block3d.z - 1) / block3d.z);
    const dim3 gridU((std::max(nx - 1, 1) + block3d.x - 1) / block3d.x,
                     (ny + block3d.y - 1) / block3d.y,
                     (nz + block3d.z - 1) / block3d.z);
    const dim3 gridV((nx + block3d.x - 1) / block3d.x,
                     (std::max(ny - 1, 1) + block3d.y - 1) / block3d.y,
                     (nz + block3d.z - 1) / block3d.z);
    const dim3 gridW((nx + block3d.x - 1) / block3d.x,
                     (ny + block3d.y - 1) / block3d.y,
                     (std::max(nz - 1, 1) + block3d.z - 1) / block3d.z);
    const int threads1d = 256;
    const int blocks1d = (cpar_host.nTot + threads1d - 1) / threads1d;

    int istep = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (!g_stop && (max_steps == 0 || istep < max_steps)) {
        ++istep;

        cal_nu_rho_kernel<<<blocks1d, threads1d>>>(d.F, d.rho, d.nu);

        advect_u_kernel<<<gridU, block3d>>>(d.u, d.v, d.w, d.rho, d.nu, d.u_star);
        advect_v_kernel<<<gridV, block3d>>>(d.u, d.v, d.w, d.rho, d.nu, d.v_star);
        advect_w_kernel<<<gridW, block3d>>>(d.u, d.v, d.w, d.rho, d.nu, d.w_star);
        CHECK_CUDA(cudaPeekAtLastError());

        for (int it = 0; it < 10; ++it) {
            solve_p_jacobi_kernel<<<gridXYZ, block3d>>>(d.u_star, d.v_star, d.w_star, d.rho, d.p, d.pt);
            CHECK_CUDA(cudaPeekAtLastError());
            std::swap(d.p, d.pt);
        }

        update_u_kernel<<<gridU, block3d>>>(d.rho, d.u_star, d.p, d.u);
        update_v_kernel<<<gridV, block3d>>>(d.rho, d.v_star, d.p, d.v);
        update_w_kernel<<<gridW, block3d>>>(d.rho, d.w_star, d.p, d.w);
        CHECK_CUDA(cudaPeekAtLastError());

        apply_bc_velocity(d.u, d.v, d.w);

        if (istep % 3 == 0) {
            run_fct_x(d.F, d.Ftd, d.u, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
            run_fct_y(d.F, d.Ftd, d.v, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
            run_fct_z(d.F, d.Ftd, d.w, d.az, d.cz, d.rp, d.rm);
        } else if (istep % 3 == 1) {
            run_fct_y(d.F, d.Ftd, d.v, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
            run_fct_z(d.F, d.Ftd, d.w, d.az, d.cz, d.rp, d.rm);
            run_fct_x(d.F, d.Ftd, d.u, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
        } else {
            run_fct_z(d.F, d.Ftd, d.w, d.az, d.cz, d.rp, d.rm);
            run_fct_x(d.F, d.Ftd, d.u, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
            run_fct_y(d.F, d.Ftd, d.v, d.ax, d.ay, d.cx, d.cy, d.rp, d.rm);
        }

        post_process_f_kernel<<<blocks1d, threads1d>>>(d.F);
        CHECK_CUDA(cudaPeekAtLastError());

        apply_bc_F(d.F);

        if (istep % nstep == 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(F_host.data(), d.F, bytes, cudaMemcpyDeviceToHost));

            std::ostringstream oss;
            oss << "output/step-" << std::setw(5) << std::setfill('0') << istep << ".vtk";
            std::cout << ">>> Exporting step-" << std::setw(5) << std::setfill('0') << istep << " result..." << std::endl;
            write_legacy_vtk(oss.str(), F_host, nx + 2, ny + 2, nz + 2);
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    std::cout << ">>> Finished at step " << istep << " in " << elapsed << " s" << std::endl;

    free_all(d);
    return 0;
}
