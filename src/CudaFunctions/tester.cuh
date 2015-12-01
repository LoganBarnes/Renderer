#ifndef TESTER_CUH
#define TESTER_CUH

#include <curand_kernel.h>

typedef unsigned int uint;
struct Shape;

extern "C"
{
    void cuda_testSamplePoint(curandState *state,
                              Shape *shape,
                              float3 *results,
                              uint numResults,
                              bool useNormals);
}

#endif // TESTER_CUH
