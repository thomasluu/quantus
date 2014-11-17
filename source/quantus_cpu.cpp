#include <cstdlib>

#include "quantus_struct.h"

template <class T>
void quantus_cpu_cleanup(quantus_comm<T> *comm)
{
    free((T *) comm->matrix);
}

template void quantus_cpu_cleanup<float>(quantus_comm<float> *);
template void quantus_cpu_cleanup<double>(quantus_comm<double> *);
