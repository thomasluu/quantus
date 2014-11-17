#include <float.h>
#include <stdio.h>

#include <curand.h>

#include "include/quantus_gamma.h"

typedef double FP;
const FP alpha = 4;

__global__ void gamma_kernel(const FP *U, FP *X, unsigned int N, quantus_comm<FP> comm) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        X[i] = quantus_gamma(U[i], &comm);
    }
}

int main()
{
    const unsigned nblocks = 19532;
    const unsigned nthreads = 512;
    const unsigned n = nblocks * nthreads;
    const size_t size = n * sizeof(FP);

    FP *d_U, *d_X;
    cudaMalloc((void **) &d_U, size);
    cudaMalloc((void **) &d_X, size);

    FP *h_U = (FP *) malloc(size);
    FP *h_X = (FP *) malloc(size);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    if (sizeof(FP) == sizeof(float)) {
        curandGenerateUniform(gen, (float *) d_U, n);
    } else {
        curandGenerateUniformDouble(gen, (double *) d_U, n);
    }

    quantus_comm<FP> comm;
    quantus_gamma_cuda_init(alpha, &comm);

    gamma_kernel<<<nblocks, nthreads>>>(d_U, d_X, n, comm);

    quantus_cuda_cleanup(&comm);

    cudaMemcpy(h_X, d_X, n * sizeof(FP), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, n * sizeof(FP), cudaMemcpyDeviceToHost);

    int digits = sizeof(FP) == sizeof(double) ? DBL_DIG : FLT_DIG;
    for (int i = 0; i < 10; i++) {
        printf("%.*e\t%.*e\n", digits, h_U[i], digits, h_X[i]);
    }

    free(h_U);
    free(h_X);

    cudaFree(d_U);
    cudaFree(d_X);

    curandDestroyGenerator(gen);
}
