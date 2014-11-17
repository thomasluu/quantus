#ifndef QUANTUS_GAMMA_H
#define QUANTUS_GAMMA_H

#include "quantus_struct.h"
#include "quantus_normal.h"
#include "quantus_cuda.h"
#include "quantus_cpu.h"

#include "wichura.h"

#include <limits>

template <class T>
void quantus_gamma_cuda_init(T shape, quantus_comm<T> *comm);

template <class T>
void quantus_gamma_cpu_init(T shape, quantus_comm<T> *comm);

__host__ __device__ float quantus_gamma(float u, quantus_comm<float> const * const comm)
{
    if (u == 1.0f) {
#ifdef __CUDA_ARCH__
        return CUDART_INF_F;
#else
        return std::numeric_limits<float>::infinity();
#endif
    }

    if (comm->magic_number == 100) {
        if (comm->param1 == 1.0f) {
            return -log1pf(-u);
        } else if (comm->param1 == 0.5f) {
            float t = erfinvf(u);
            return t * t;
        } else {
            return expf((logf(u) + comm->cache2) / comm->param1);
        }
    }

    if (comm->magic_number == 0) {
        if (u < comm->cache1) {
            return expf((logf(u) + comm->cache2) / comm->param1);
        }
    }

    float v;
#ifdef __CUDA_ARCH__
    v = quantus_normal(u);
#else
    v = ppnd7(u);
#endif

    float p, x;
    v = modf(ldexp(v, comm->spacing) - comm->offset, &p);
    x = v;
    v = p;
    p = comm->matrix[(int)v];
    if (v < 0) {
        return 0;
    }
    float yk2 = 0;
    float yk1 = 0;
    float *row_ptr;
    for (int row = 1; row < comm->height - 1; row++) {
        yk2 = yk1;
        yk1 = p;
#ifdef __CUDA_ARCH__
        row_ptr = (float*) ((char*)comm->matrix + row * comm->pitch);
#else
        row_ptr = (float*) (comm->matrix + (int)row * comm->width);
#endif
        p = 2.0f * x * yk1 - yk2 + row_ptr[(int)v];
    }
#ifdef __CUDA_ARCH__
    row_ptr = (float*) ((char*)comm->matrix + (comm->height - 1) * comm->pitch);
#else
    row_ptr = (float*) (comm->matrix + (int)(comm->height - 1) * comm->width);
#endif
    p = 0.5f * row_ptr[(int)v] + p * x - yk1;

    if (comm->magic_number == 0) {
        p = exp(p);
    }

    return p;
}

__host__ __device__ double quantus_gamma(double u, quantus_comm<double> const * const comm)
{
    if (u == 1.0) {
#ifdef __CUDA_ARCH__
        return CUDART_INF;
#else
        return std::numeric_limits<double>::infinity();
#endif
    }

    if (comm->magic_number == 100) {
        if (comm->param1 == 1.0) {
            return -log1p(-u);
        } else if (comm->param1 == 0.5) {
            double t = erfinv(u);
            return t * t;
        } else {
            return exp((log(u) + comm->cache2) / comm->param1);
        }
    }
 
    if (comm->magic_number == 0) {
        if (u < comm->cache1) {
            return exp((log(u) + comm->cache2) / comm->param1);
        }
    }

    double v;
#ifdef __CUDA_ARCH__
    v = quantus_normal(u);
#else
    v = ppnd16(u);
#endif

    double p, x;
    v = modf(ldexp(v, comm->spacing) - comm->offset, &p);
    x = v;
    v = p;
    p = comm->matrix[(int)v];
    if (v < 0) {
        return 0;
    }
    double yk2 = 0;
    double yk1 = 0;
    double *row_ptr;
    for (int row = 1; row < comm->height - 1; row++) {
        yk2 = yk1;
        yk1 = p;
#ifdef __CUDA_ARCH__
        row_ptr = (double*) ((char*)comm->matrix + row * comm->pitch);
#else
        row_ptr = (double*) (comm->matrix + (int)row * comm->width);
#endif
        p = 2.0 * x * yk1 - yk2 + row_ptr[(int)v];
    }
#ifdef __CUDA_ARCH__
    row_ptr = (double*) ((char*)comm->matrix + (comm->height - 1) * comm->pitch);
#else
    row_ptr = (double*) (comm->matrix + (int)(comm->height - 1) * comm->width);
#endif
    p = 0.5 * row_ptr[(int)v] + p * x - yk1;

    if (comm->magic_number == 0) {
        p = exp(p);
    }

    return p;
}

#endif
