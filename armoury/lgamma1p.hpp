#ifndef LGAMMA1P_HPP
#define LGAMMA1P_HPP

#include <boost/math/special_functions/zeta.hpp>

namespace armoury{

/*
 * A&S 6.1.33
 */
template <class T>
T lgamma1p(T x)
{
  if (fabs(x) >= .5) {
    return lgamma(1 + x);
  }

  typedef boost::math::policies::policy<
      boost::math::policies::promote_double<false>
      > my_policy; // don't use long double internally

  T sum = 0;
  T pow_x = x * x;
  for (int n = 2; n < 1000; n++) {
    T temp = (boost::math::zeta(n, my_policy()) - 1) * pow_x / n;
    T old_sum = sum;
    if (n % 2 == 0) {
      sum += temp;
    } else {
      sum -= temp;
    }
    if (fabs(1 - old_sum / sum) <= std::numeric_limits<T>::epsilon()) {
      break;
    }
    pow_x *= x;
  }

  //return -boost::math::log1p(x, my_policy()) + x * (1 - boost::math::constants::euler<T>()) + sum;
  return -boost::math::log1pmx(x, my_policy()) - x * boost::math::constants::euler<T>() + sum;
}

} // namespace armoury

#endif
