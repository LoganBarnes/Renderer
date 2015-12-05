#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH

#include <curand_kernel.h>

typedef unsigned int uint;
struct Shape;

extern "C"
{
    /*
     * from 'pathTracer.cu'
     */
    void cuda_tracePath(cudaSurfaceObject_t surface,
                        float *scaleViewInv,
                        Shape *shapes,
                        uint numShapes,
                        Shape *areaLights,
                        uint numAreaLights,
                        dim3 texDim,
                        curandState *randState,
                        bool debugEmit,
                        bool debugDirect,
                        bool debugIndirect,
                        int bounceLimit = 1000,
                        float scale = 1.f);
}

#endif // CUDA_RENDER_CUH
