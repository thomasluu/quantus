#include <iostream>
#include <limits>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>

#include <armoury/taylor.hpp>
#include <armoury/norm2expgam.hpp>
#include <armoury/norm2gam.hpp>
#include <armoury/boost_policy.hpp>

template <class T>
struct BackwardError
{
    BackwardError(T alpha) : gam(boost::math::gamma_distribution<T, my_policy>(alpha)) {}
    T operator()(T v, T Q)
    { 
        T exp_Q = exp(Q);
        if (isinf(exp_Q) || isnan(exp_Q)) return std::numeric_limits<T>::max();
        T exp_gam_cdf = cdf(gam, exp_Q);
        T norm_cdf = cdf(norm, v);
        return fabs(1 - exp_gam_cdf / norm_cdf);
    }
private:
    boost::math::gamma_distribution<T, my_policy> gam;
    boost::math::normal_distribution<T, my_policy> norm;
};

template <typename TP, typename WP, typename HP>
std::vector<std::vector<TP> > master_quantus_gamma_coeffs(int magic_number, const TP shape, TP &acc, double &h, int &m)
{
    std::vector<std::pair<WP, std::vector<WP> > > polys;
    uint16_t order = (sizeof(TP) == sizeof(double)) ? 20 : 10;
    WP h_initial = (sizeof(TP) == sizeof(double)) ? .125 : .25;
    WP h_min = (sizeof(TP) == sizeof(double)) ? 0.0625 : .125;
    // TODO: Add argument to specify relaxed tolerance?
    WP fwd_tol = (shape < 1) ? 50 * std::numeric_limits<TP>::epsilon() : 50 * std::numeric_limits<TP>::epsilon();
    WP bwd_tol = (shape > 1) ? 50 * std::numeric_limits<TP>::epsilon() : 50 * std::numeric_limits<TP>::epsilon();
    WP h_got;
    WP tol_got;
    if (magic_number == 0) {
        armoury::norm2expgam_ivp<HP, WP, TP> ivp(shape/*, 64*/); // TODO: Add argument to specify RNG bits
        armoury::norm2expgam_ad<WP> ad(shape);
        polys = armoury::taylor<WP>(ivp, ad, order, h_initial, h_min, fwd_tol, bwd_tol, BackwardError<WP>(shape), h_got, tol_got);
    } else {
        armoury::norm2gam_ivp<HP, WP, TP> ivp(shape/*, 64*/);
        armoury::norm2gam_ad<WP> ad(shape);
        polys = armoury::taylor<WP>(ivp, ad, order, h_initial, h_min, fwd_tol, bwd_tol, BackwardError<WP>(shape), h_got, tol_got);
    }

    acc = tol_got;
    h = h_got;
    m = (!polys.empty()) ? polys[0].first / h_got : 0;

    std::vector<std::vector<TP> > polys_tp(polys.size());
    for (int i = 0; i < polys.size(); i++) {
        std::vector<WP> poly_wp = polys[i].second;
        for (int j = 0; j < poly_wp.size(); j++) {
            polys_tp[i].push_back(poly_wp[j]);
        }
    }
    return polys_tp;
}

template <typename FP>
std::vector<std::vector<FP> > quantus_gamma_coeffs(int magic_number, const FP shape, FP &acc, double &h, int &m);

template <>
std::vector<std::vector<float> > quantus_gamma_coeffs<float>(int magic_number, const float shape, float &acc, double &h, int &m)
{
    return master_quantus_gamma_coeffs<float, double, double>(magic_number, shape, acc, h, m);
}

template <>
std::vector<std::vector<double> > quantus_gamma_coeffs<double>(int magic_number, const double shape, double &acc, double &h, int &m)
{
    return master_quantus_gamma_coeffs<double, double, boost::multiprecision::number<boost::multiprecision::cpp_dec_float<20> > >(magic_number, shape, acc, h, m);
}

template std::vector<std::vector<float> > quantus_gamma_coeffs<float>(int, const float, float &, double &, int &);
template std::vector<std::vector<double> > quantus_gamma_coeffs<double>(int, const double, double &, double &, int &);
