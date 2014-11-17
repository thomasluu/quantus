#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <armoury/lgamma1p.hpp>

#include "quantus_cuda.h"
#include "quantus_cuda_init.cpp"

#include "quantus_cpu.h"
#include "quantus_cpu_init.cpp"

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/constants/constants.hpp>

template <typename FP>
std::vector<std::vector<FP> > quantus_gamma_coeffs(int magic_number, const FP shape, FP &acc, double &h, int &m);

template <class T>
class quantus_gamma_distribution
{
    T shape;
public:

    quantus_gamma_distribution(T shape) : shape(shape)
    {
        if (shape <= 0) {
            throw std::domain_error("shape <= 0");
        }
    }

    std::vector<T> parameters()
    {
        std::vector<T> param;
        param.push_back(shape);
        return param;
    }

    void operator()(int &magic_number, std::vector<std::vector<T> > &matrix, int &spacing, int &offset, T &cache1, T &cache2)
    {
        if (shape == 1 || shape == 0.5) {
            magic_number = 100;
            return;
        }

        magic_number = shape < 1000 ? 0 : 1;

        // transformation approximation

        double h;
        int m;
        T expected_accuracy;
        std::vector<std::vector<T> > c = quantus_gamma_coeffs<T>(magic_number, shape, expected_accuracy, h, m);

        if (c.empty()) {
            magic_number = 100;
            cache1 = 1;
            cache2 = shape * tgamma(shape);
            return;
        }

        matrix = c;
        spacing = -log(h) / boost::math::constants::ln_two<double>();
        offset = m;

        if (magic_number == 0) {
            if (shape < 100) {
                T eps = exp2(-std::numeric_limits<T>::digits);
                cache1 = pow(-log1p(-eps), shape) / (shape * tgamma(shape));
                cache2 = armoury::lgamma1p((double)shape);
            } else {
                cache1 = 0;
            }
        }
    }
};

template <class T>
void quantus_gamma_cuda_init(T shape, quantus_comm<T> *comm)
{
    quantus_gamma_distribution<T> dist(shape);
    quantus_cuda_init(dist, comm);
}

template void quantus_gamma_cuda_init<float>(float, quantus_comm<float> *);
template void quantus_gamma_cuda_init<double>(double, quantus_comm<double> *);

template <class T>
void quantus_gamma_cpu_init(T shape, quantus_comm<T> *comm)
{
    quantus_gamma_distribution<T> dist(shape);
    quantus_cpu_init(dist, comm);
}

template void quantus_gamma_cpu_init<float>(float, quantus_comm<float> *);
template void quantus_gamma_cpu_init<double>(double, quantus_comm<double> *);
