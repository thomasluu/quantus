#include <algorithm>
#include <vector>

#include "quantus_struct.h"

template <class T, class Distribution>
void quantus_cpu_init(Distribution dist, quantus_comm<T> *comm)
{
    std::vector<T> param = dist.parameters();

    if (param.size() > 0) {
        comm->param1 = param[0];
    }

    if (param.size() > 1) {
        comm->param2 = param[1];
    }

    int magic_number;
    std::vector<std::vector<T> > matrix;
    int spacing;
    int offset;
    T cache1;
    T cache2;

    dist(magic_number, matrix, spacing, offset, cache1, cache2);

    comm->cache1 = cache1;
    comm->cache2 = cache2;

    comm->magic_number = magic_number;
    if (magic_number == -1 || magic_number == 100) {
        return;
    }

    if (!matrix.empty()) {
        for (int i = 0; i < matrix.size(); i++) { // iterating over each polynomial
            std::reverse(matrix[i].begin(), matrix[i].end());
        }

        int width = matrix.size(); // number of polynomials
        int height = matrix[0].size(); // number of coefficients per polynomial

        T *h_matrix;
        h_matrix = (T *) malloc(width * height * sizeof(T));

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                h_matrix[j * width + i] = matrix[i][j];
            }
        }

        comm->matrix = h_matrix;

        comm->height = height;
        comm->width = width;
        comm->spacing = spacing;
        comm->offset = offset;
    } else {
        comm->matrix = NULL;
    }
}
