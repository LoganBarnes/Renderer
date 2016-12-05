#ifndef TESTER_CUH
#define TESTER_CUH

#include <curand_kernel.h>

typedef unsigned int uint;

struct Shape;
struct SurfaceElement;
struct Ray;

extern "C"
{
    void cuda_testSamplePoint(curandState *state,
                              Shape *shape,
                              float3 *results,
                              uint numResults,
                              bool useNormals);

    void cuda_testSphereIntersect(Shape *shape, SurfaceElement *surfel, Ray *ray);
}

#endif // TESTER_CUH
