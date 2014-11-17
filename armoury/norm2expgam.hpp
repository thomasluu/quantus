#ifndef ARMOURY_NORM2EXPGAM_HPP
#define ARMOURY_NORM2EXPGAM_HPP

#include <vector>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/gamma.hpp>

#include <armoury/boost_policy.hpp>

#define KAHAN

namespace armoury{

template <typename FP>
class norm2expgam_ad
{
public:
    norm2expgam_ad(const FP &alpha);
    std::vector<FP> taylor_coeff(const FP &, int);
    void reset(const std::vector<FP> &);
private:
    FP alpha;
    std::vector<FP> lut[8];
    FP V0(const FP &, int);
    FP V1(const FP &, int);
    FP V2(const FP &, int);
    FP V3(const FP &, int);
    FP V4(const FP &, int);
    FP V5(const FP &, int);
    FP Q0(const FP &, int);
    FP Q1(const FP &, int);
};

template <typename FP>
norm2expgam_ad<FP>::norm2expgam_ad(const FP &alpha) : alpha(alpha) {}

template <typename FP>
std::vector<FP> norm2expgam_ad<FP>::taylor_coeff(const FP &v, int n)
{
    std::vector<FP> result;
    result.push_back(Q0(v, n));
    result.push_back(Q1(v, n));
    return result;
}

template <typename FP>
void norm2expgam_ad<FP>::reset(const std::vector<FP> &Q)
{
    for (int i = 0; i < 8; i++) lut[i].clear();
    lut[6].push_back(Q[0]);
    lut[7].push_back(Q[1]);
}

template <typename FP>
FP norm2expgam_ad<FP>::V0(const FP &v, int k)
{
    if (k < lut[0].size()) return lut[0][k];

    FP result;
    switch (k) {
        case 0: result = v; break;
        case 1: result = 1; break;
        default: result = 0; break;
    }
    lut[0].push_back(result);
    return result;
}

template <typename FP>
FP norm2expgam_ad<FP>::V1(const FP &v, int k)
{
    if (k < lut[1].size()) return lut[1][k];

    FP result;
    if (k == 0) {
        result = exp(Q0(v, 0));
    } else {
        result = 0;
        for (int j = 0; j < k; j++) {
            result += (k - j) * V1(v, j) * Q0(v, k - j);
        }
        result /= k;
    }
    lut[1].push_back(result);
    return result;
}

template <typename FP>
FP norm2expgam_ad<FP>::V2(const FP &v, int k)
{
    if (k < lut[2].size()) return lut[2][k];

    FP result = V1(v, k);
    if (k == 0) result -= alpha;
    lut[2].push_back(result);
    return result;
}

template <typename FP>
FP norm2expgam_ad<FP>::V3(const FP &v, int k)
{
    if (k < lut[3].size()) return lut[3][k];

    FP sum = 0;
#ifndef KAHAN
    for (int j = 0; j <= k; j++) {
        sum += V2(v, j) * Q1(v, k - j);
    }
#else
    FP c = 0;
    for (int j = 0; j <= k; j++) {
        FP y = V2(v, j) * Q1(v, k - j) - c;
        FP t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif
    lut[3].push_back(sum);
    return sum;
}

template <typename FP>
FP norm2expgam_ad<FP>::V4(const FP &v, int k)
{
    if (k < lut[4].size()) return lut[4][k];

    FP result = V3(v, k) - V0(v, k);
    lut[4].push_back(result);
    return result;
}

template <typename FP>
FP norm2expgam_ad<FP>::V5(const FP &v, int k)
{
    if (k < lut[5].size()) return lut[5][k];

    FP sum = 0;
#ifndef KAHAN
    for (int j = 0; j <= k; j++) {
        sum += Q1(v, j) * V4(v, k - j);
    }
#else
    FP c = 0;
    for (int j = 0; j <= k; j++) {
        FP y = Q1(v, j) * V4(v, k - j);
        FP t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif
    lut[5].push_back(sum);
    return sum;
}

template <typename FP>
FP norm2expgam_ad<FP>::Q0(const FP &v, int k)
{
    if (k < lut[6].size()) return lut[6][k];

    FP result = Q1(v, k - 1) / k;
    lut[6].push_back(result);
    return result;
}

template <typename FP>
FP norm2expgam_ad<FP>::Q1(const FP &v, int k)
{
    if (k < lut[7].size()) return lut[7][k];

    FP result = V5(v, k - 1) / k;
    lut[7].push_back(result);
    return result;
}

template <typename FP>
FP u_upper(FP x, FP alpha)
{
  return pow(-log1p(-x), alpha) / (alpha * tgamma(alpha));
}

template <typename HP, typename WP, class TP>
struct norm2expgam_ivp
{
    norm2expgam_ivp(TP alpha, uint16_t rng_bits = 32) : alpha(alpha), rng_bits(rng_bits) {}
    WP initial_point()
    {
        TP eps = exp2(-std::numeric_limits<TP>::digits);
        TP u_eps = u_upper(std::numeric_limits<TP>::epsilon(), alpha);
        if (u_eps >= 1) {
            u_eps = 1;
        }

        boost::math::gamma_distribution<TP, my_policy> gam(alpha);
        TP gamcdf = boost::math::cdf(gam, std::numeric_limits<TP>::denorm_min());
        if (gamcdf == 1) {
            u_eps = 1;
        }

        double u = fmax(exp2(-rng_bits), u_eps);
        boost::math::normal_distribution<double, my_policy> norm;
        WP v = boost::math::quantile(norm, u);
        return v;
    }
    std::vector<WP> initial_condition(WP v)
    {
        boost::math::normal_distribution<WP, my_policy> norm;
        WP normcdf = boost::math::cdf(norm, v);

        if (normcdf < 0.9) {
            boost::math::gamma_distribution<WP, my_policy> gam(alpha);
            WP Q0 = log(boost::math::quantile(gam, normcdf));
            std::vector<WP> Q;
            Q.push_back(Q0);
            Q.push_back(boost::math::pdf(norm, v) * exp((exp(Q0) - Q0 * alpha) + lgamma(alpha)));
            return Q;
        } else {
            boost::math::normal_distribution<HP, my_policy> norm;
            HP normcdf = boost::math::cdf(norm, v);

            boost::math::gamma_distribution<HP, my_policy> gam(alpha);
            HP Q0 = log(boost::math::quantile(gam, normcdf));
            std::vector<WP> Q;
            Q.push_back(boost::math::tools::real_cast<WP>(Q0));
            Q.push_back(boost::math::tools::real_cast<WP>(boost::math::pdf(norm, v) * exp((exp(Q0) - Q0 * alpha) + lgamma(alpha))));
            return Q;
        }
    }
    WP end_point()
    {
        double u = 1 - exp2(-fmin(std::numeric_limits<TP>::digits, rng_bits));
        boost::math::normal_distribution<double, my_policy> norm;
        WP v = boost::math::quantile(norm, u);
        return v;
    }
    std::vector<WP> operator()(const WP &v, const std::vector<WP> &Q)
    {
        std::vector<WP> dQdv;
        dQdv.push_back(Q[1]);
        dQdv.push_back(Q[1] * ((exp(Q[0]) - alpha) * Q[1] - v));
        return dQdv;
    }
private:
    const TP alpha;
    const uint16_t rng_bits;
};

} // namespace armoury

#endif
