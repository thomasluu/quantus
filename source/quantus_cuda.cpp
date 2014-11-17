#include <cuda_runtime.h>

#include "quantus_struct.h"

template <class T>
void quantus_cuda_cleanup(quantus_comm<T> *comm)
{
    cudaFree((T *) comm->matrix);
}

template void quantus_cuda_cleanup<float>(quantus_comm<float> *);
template void quantus_cuda_cleanup<double>(quantus_comm<double> *);
