#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h" // includes helper_math.h
#include "shared/renderObjects.hpp"
#include "intersections.cu"


extern "C"
{

    // part one of the dft
    __global__
    void tracePath_kernel(cudaSurfaceObject_t surfObj,
                          float4 *scaleViewInvEye,
                          Shape *shapes,
                          uint numShapes,
                          Luminaire *luminaires,
                          uint numLuminaires,
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
            ray.orig = make_float3(scaleViewInvEye[4]);
            ray.dir = make_float3(scaleViewInvEye * farPoint);
            ray.dir = normalize(ray.dir - ray.orig);

            Shape shape;
            float4 n;
            int index = intersectWorld(ray, shapes, numShapes, n, shape, -1);
            float4 result;

            if (index >= 0)
            {
                result = shape.mat.color;
            }
            else
            {
                result = make_float4(0, 0, 0, 1);
            }

//            if (x == texDim.x / 2 && y == texDim.y / 2)
//            {
//                if (numShapes > 0)
//                {
//                    Shape shape = shapes[0];
//                    float4 color = shape.mat.color;
//                    printf("Shape: %d, color: (%.2f, %.2f, %.2f, %.2f)\n", (int)shape.type, color.x, color.y, color.z, color.w);
//                    printf("Ray: orig(%.2f, %.2f, %.2f)::dir(%.2f, %.2f, %.2f); Index: %d\n",
//                           ray.orig.x, ray.orig.y, ray.orig.z,
//                           ray.dir.x, ray.dir.y, ray.dir.z, index);
//                }
//            }
            // Write to output surface
            surf2Dwrite(result, surfObj, x * sizeof(float4), y);

        }
    }

    void cuda_tracePath(cudaSurfaceObject_t surface,
                        float *scaleViewInvEye,
                        Shape *shapes,
                        uint numShapes,
                        Luminaire *luminaires,
                        uint numLuminaires,
                        dim3 texDim)
    {
        dim3 thread(32, 32);
        dim3 block(texDim.x / thread.x, texDim.y / thread.y);
        tracePath_kernel<<< block, thread >>>(surface,
                                              (float4 *)scaleViewInvEye,
                                              shapes,
                                              numShapes,
                                              luminaires,
                                              numLuminaires,
                                              texDim);
    }
}
