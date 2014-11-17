#ifndef ARMOURY_BOOST_POLICY
#define ARMOURY_BOOST_POLICY

using namespace boost::math::policies;

typedef policy<
    overflow_error<ignore_error>,
#if 1
    promote_float<false>,
    promote_double<false>
#else
    promote_float<true>,
    promote_double<true>
#endif
> my_policy;

#endif
