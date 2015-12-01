#include "helper_cuda.h"
#include "helper_grid.h"
#include "random_kernel.cu"
#include "renderObjects.hpp"

extern "C"
{
    __global__
    void testSamplePoint_kernel(curandState *state, Shape *shape, float3 *results, uint numResults, bool useNormals)
    {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < numResults)
        {
            SurfaceElement surfel = samplePoint(state, idx, *shape);
            if (useNormals)
                results[idx] = surfel.normal;
            else
                results[idx] = surfel.point;
        }
    }

    void cuda_testSamplePoint(curandState *state, Shape *shape, float3 *results, uint numResults, bool useNormals)
    {
        dim3 thread(64);
        dim3 block(1);
        computeGridSize(numResults, thread.x, block.x, thread.x);

        testSamplePoint_kernel<<<block, thread>>>(state, shape, results, numResults, useNormals);
    }

}
