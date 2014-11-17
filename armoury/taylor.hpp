#ifndef ARMOURY_TAYLOR_HPP
#define ARMOURY_TAYLOR_HPP

#include <vector>
#include <map>
#include <iterator>

#include <omp.h>

#include <boost/math/tools/polynomial.hpp>

#if 1
template <class T>
T evaluate_chebyshev(const std::vector<T>& a, const T& x)
{
   return boost::math::tools::evaluate_chebyshev(a, x);
}
#endif

namespace armoury{

/*
 * Eq. 3.33
 */
template <class T>
T th(int r, int k)
{
  if (r == k) {
    return exp2(T(1) - k);
  }
  int j = (r - k) / 2;
  if ((r - k) % 2 == 1) {
    return 0;
  } else {
    return (k + T(2) * j) * (k + T(2) * j - T(1)) / (T(4) * j * (k + j)) * th<T>(k + T(2) * (j - T(1)), k);
  }
}

/*
 * Eq. 3.31 and 3.32
 */
template <class T>
T cheb_coeff(int k, int n, std::vector<T> tay_coeff, T h)
{
  T sum = 0;
  T power = pow(h, k);
  for (int r = k; r <= n; r++) {
    sum += tay_coeff[r] * power * th<T>(r, k);
    power *= h;
  }
  return sum;
}

template <class T, class IVP, class AD>
std::vector<T>
coeff(IVP ivp, AD ad, uint16_t n, T x)
{
    std::vector<T> y0 = ivp.initial_condition(x);
    AD my_ad(ad);
    my_ad.reset(y0);

    std::vector<std::vector<T> > c(y0.size());
    for (int j = 0; j <= n; j++) {
        std::vector<T> res = my_ad.taylor_coeff(x, j);
        for (int k = 0; k < res.size(); k++) {
            c[k].push_back(res[k]);
        }
    }

    return c[0];
}

template <class T, class IVP, class BackwardError>
std::vector<std::pair<T, std::vector<T> > >
check(IVP ivp, const std::map<T, std::vector<T> > sol, uint16_t n, T h, T initial_point, T tol_fwd, T tol_bwd, BackwardError backward_error, T &tol)
{
    typename std::map<T, std::vector<T> >::const_iterator it = sol.begin();
    for (; it != sol.end(); it++) {
        if (it->first + h > initial_point) break;
    }

    std::vector<std::pair<T, std::vector<T> > > tay_pairs(it, sol.end());
    std::vector<std::pair<T, std::vector<T> > > cheb_pairs(it, sol.end());

    for (int m = 1; m <= n; m++) {
        T fwd_err = 0;
        T bwd_err = 0;
        for (int i = 0; i < tay_pairs.size(); i++) {
            T x = tay_pairs[i].first;
            std::vector<T> c = tay_pairs[i].second;
            c.resize(m + 1);

            std::vector<T> cheb;
            for (int nn = 0; nn <= m; nn++) {
                cheb.push_back( cheb_coeff(nn, m, c, h) );
            }

            T test = evaluate_chebyshev<T>(cheb, 1);
            T target = (i < tay_pairs.size() - 1) ? tay_pairs[i + 1].second[0] : ivp.initial_condition(x + h)[0];
            T accuracy = fabs(1 - test / target);
            if (fwd_err < accuracy) {
                fwd_err = accuracy;
            }

            accuracy = backward_error(x + h, test);
            if (bwd_err < accuracy) {
                bwd_err = accuracy;
            }
        }

        if ((fwd_err < tol_fwd) || (fwd_err < tol_fwd && bwd_err < tol_bwd)) {
            for (int i = 0; i < tay_pairs.size(); i++) {
                T x = tay_pairs[i].first;
                tay_pairs[i].second.resize(m + 1);

                std::vector<T> cheb;
                for (int nn = 0; nn <= m; nn++) {
                    cheb.push_back( cheb_coeff(nn, m, tay_pairs[i].second, h) );
                }
                cheb_pairs[i].second = cheb;
            }
            tol = fwd_err;
            return cheb_pairs;
        }
    }

    return std::vector<std::pair<T, std::vector<T> > >();
}

template <class T, class IVP, class AD, class BackwardError>
std::vector<std::pair<T, std::vector<T> > >
taylor(IVP ivp, AD ad, uint16_t n, T h, T h_min, T tol_fwd, T tol_bwd, BackwardError backward_error, T &h_actual, T &tol_actual)
{
    T initial_point = ivp.initial_point();
    T end_point = ivp.end_point();
    if (isnan(initial_point) || end_point <= initial_point) {
        h_actual = 0;
        tol_actual = 0;
        return std::vector<std::pair<T, std::vector<T> > >();
    }

    T working_initial_point = floor(initial_point / h) * h;

    std::map<T, std::vector<T> > sol; // point and series expansion pairs

    /*
     * populate sol
     */

    std::vector<std::pair<T, std::vector<T> > > temp;
    temp.resize(1 + floor((end_point - working_initial_point) / h));

    #pragma omp parallel for
    for (int i = 0; i < temp.size(); i++) {
        T x = working_initial_point + i * h;
        std::vector<T> c = coeff(ivp, ad, n, x);
        temp[i] = std::pair<T, std::vector<T> >(x, c);
    }
    sol.insert(temp.begin(), temp.end());

    T tol;
    std::vector<std::pair<T, std::vector<T> > > table = check(ivp, sol, n, h, initial_point, tol_fwd, tol_bwd, backward_error, tol);
    if (!table.empty()) {
        h_actual = h;
        tol_actual = tol;
        return table;
    }

    int times = log2(h) - log2(h_min);
    for (int j = 0; j < times; j++) {
        h /= 2;
        temp.resize(ceil(floor((end_point - working_initial_point) / h) / 2));

        #pragma omp parallel for
        for (int i = 0; i < temp.size(); i++) {
            T x = working_initial_point + (i + i + 1) * h;
            std::vector<T> c = coeff(ivp, ad, n, x);
            temp[i] = std::pair<T, std::vector<T> >(x, c);
        }
        sol.insert(temp.begin(), temp.end());

        std::vector<std::pair<T, std::vector<T> > > table = check(ivp, sol, n, h, initial_point, tol_fwd, tol_bwd, backward_error, tol);
        if (!table.empty()) {
            h_actual = h;
            tol_actual = tol;
            return table;
        }
    }

    // XXX: Refactor!
    typename std::map<T, std::vector<T> >::iterator it = sol.begin();
    for (; it != sol.end(); it++) {
        if (it->first + h > initial_point) break;
    }

    std::vector<std::pair<T, std::vector<T> > > tay_pairs(it, sol.end());
    std::vector<std::pair<T, std::vector<T> > > cheb_pairs(it, sol.end());

    T fwd_err = 0;
    T bwd_err = 0;
    for (int i = 0; i < tay_pairs.size(); i++) {
        T x = tay_pairs[i].first;
        std::vector<T> c = tay_pairs[i].second;

        std::vector<T> cheb;
        for (int nn = 0; nn < c.size(); nn++) {
            cheb.push_back( cheb_coeff(nn, c.size(), c, h) );
        }
        cheb_pairs[i].second = cheb;

        T test = evaluate_chebyshev<T>(cheb, 1);
        T target = (i < tay_pairs.size() - 1) ? tay_pairs[i + 1].second[0] : ivp.initial_condition(x + h)[0];
        T accuracy = fabs(1 - test / target);
        if (fwd_err < accuracy) {
            fwd_err = accuracy;
        }

        accuracy = backward_error(x + h, test);
        if (bwd_err < accuracy) {
            bwd_err = accuracy;
        }
    }
 
    h_actual = h;
    tol_actual = fwd_err;
    return cheb_pairs;
}

} // namespace armoury

#endif
