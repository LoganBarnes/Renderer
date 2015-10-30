#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "renderObjects.hpp" // includes helper_math.h

extern "C"
{

    // part one of the dft
    __global__
    void tracePath_kernel(cudaSurfaceObject_t surfObj,
                                     float4 *scaleViewInvEye,
                                     dim3 texDim)
    {
        // Calculate surface coordinates
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < texDim.x && y < texDim.y)
        {
            float4 data;

            // Read from input surface
            surf2Dread(&data,  surfObj, x * sizeof(float4), y);

            float2 coords = make_float2((x + 0.5f) / texDim.x, (y + 0.5f) / texDim.y);
            coords = (coords * 2.f) - 1.f; // screen space

            float4 farPoint = make_float4(coords.x, coords.y, -1, 1);

            Ray ray;
            ray.orig = make_float3(scaleViewInvEye[3]);
            ray.dir = make_float3(scaleViewInvEye * farPoint);
            ray.dir = normalize(ray.dir - ray.orig);

            float4 result = make_float4(ray.dir, 1);
            result = clamp(result, 0, 1);

            // Write to output surface
            surf2Dwrite(result, surfObj, x * sizeof(float4), y);
        }
    }

    void cuda_tracePath(cudaSurfaceObject_t surface, float *scaleViewInvEye, dim3 texDim)
    {
        dim3 thread(32, 32);
        dim3 block(texDim.x / thread.x, texDim.y / thread.y);
        tracePath_kernel<<< block, thread >>>(surface, (float4 *)scaleViewInvEye, texDim);
    }
}
