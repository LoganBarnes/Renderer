#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h" // includes helper_math.h
#include "renderObjects.hpp"
#include "intersections.cu"
#include "random_kernel.cu"

__device__ const bool EMIT = true;
__device__ const bool DIRECT = true;
__device__ const bool INDIRECT = true;

__device__ const float BUMP_VAL = 0.0001f;
__device__ const float PI_F = 3.141592653539f;

extern "C"
{

    /**
     * @brief estimateDirectLightFromAreaLights
     * @param surfel
     * @param ray
     * @param areaLights
     * @param numAreaLights
     * @return
     */
    __device__
    Radiance3 estimateDirectLightFromAreaLights(SurfaceElement surfel,
                                                Ray ray,
                                                Shape *shapes,
                                                uint numShapes,
                                                Shape *areaLights,
                                                uint numAreaLights,
                                                curandState *randState,
                                                int id)
    {
        Radiance3 L_o = make_float3(0.f);

        for (uint l = 0; l < numAreaLights; ++l)
        {
            SurfaceElement lightSurfel = samplePoint(randState, id, areaLights[l]);

            Ray r;
            r.orig = surfel.point + surfel.normal * BUMP_VAL;
            r.dir = normalize(lightSurfel.point - r.orig);
            SurfaceElement intersection;
            if (intersectWorld(r, shapes, numShapes, intersection, surfel.index) &&
                    intersection.index == lightSurfel.index)
            {
                float3 w_i = lightSurfel.point - surfel.point;
                const float distance2 = dot(w_i, w_i);
                w_i /= sqrt(distance2);

                L_o += surfel.material.color ;//* // should calc BDSF
//                        (lightSurfel.material.emitted / PI_F) *
//                        max(0.f, dot(w_i, surfel.normal)) *
//                        max(0.f, dot(-w_i, lightSurfel.normal / distance2));
            }
        }

        return L_o;
    }


    /**
     * @brief pathTrace
     * @param ray
     * @param coeff
     * @param shapes
     * @param numShapes
     * @param areaLights
     * @param numAreaLights
     * @param isEyeRay
     * @return
     */
    __device__
    Radiance3 pathTrace(Ray &ray,
                        float &coeff,
                        Shape *shapes,
                        uint numShapes,
                        Shape *areaLights,
                        uint numAreaLights,
                        bool isEyeRay,
                        curandState *randState,
                        int id)
    {
        Radiance3 L_o = make_float3(0.f);

        SurfaceElement surfel;
        if (intersectWorld(ray, shapes, numShapes, surfel, -1))
        {
            if (isEyeRay && EMIT)
                L_o += coeff * surfel.material.emitted;

            if (!isEyeRay || DIRECT)
            {
                L_o += coeff * estimateDirectLightFromAreaLights(surfel,
                                                                 ray,
                                                                 shapes,
                                                                 numShapes,
                                                                 areaLights,
                                                                 numAreaLights,
                                                                 randState,
                                                                 id);
            }

            if (!isEyeRay || INDIRECT)
            {
                // TODO: indirect illumination
                ray.isValid = false;
            }

            L_o += coeff * surfel.material.color;
        }
        else
            ray.isValid = false;

        return L_o;
    }


    /**
     * @brief tracePath_kernel
     * @param surfObj
     * @param scaleViewInvEye
     * @param shapes
     * @param numShapes
     * @param areaLights
     * @param numAreaLights
     * @param texDim
     * @param randState
     */
    __global__
    void tracePath_kernel(cudaSurfaceObject_t surfObj,
                          float4 *scaleViewInvEye,
                          Shape *shapes,
                          uint numShapes,
                          Shape *areaLights,
                          uint numAreaLights,
                          dim3 texDim,
                          curandState *randState)
    {
        // Calculate surface coordinates
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < texDim.x && y < texDim.y)
        {
            uint id = y * texDim.x + x;
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
            ray.isValid = true;

            float coeff = 1.f;
            Radiance3 radiance = make_float3(0.f);

            while (ray.isValid)
            {
                radiance += pathTrace(ray,
                                      coeff,
                                      shapes,
                                      numShapes,
                                      areaLights,
                                      numAreaLights,
                                      true,
                                      randState,
                                      id);
            }

//            // Temp randomness
//            float3 randomness = randCosHemi(randState, id);
//            radiance *= randomness;

            float scale = 1.f;
            float4 result = make_float4(radiance * (1.f / scale), 1.f);

            // Write to output surface
            surf2Dwrite(result, surfObj, x * sizeof(float4), y);

        }
    }


    /**
     * @brief cuda_tracePath
     * @param surface
     * @param scaleViewInvEye
     * @param shapes
     * @param numShapes
     * @param areaLights
     * @param numAreaLights
     * @param texDim
     * @param randState
     */
    void cuda_tracePath(cudaSurfaceObject_t surface,
                        float *scaleViewInvEye,
                        Shape *shapes,
                        uint numShapes,
                        Shape *areaLights,
                        uint numAreaLights,
                        dim3 texDim,
                        curandState *randState)
    {
        dim3 thread(32, 32);
        dim3 block(static_cast<unsigned long>(std::ceil(texDim.x / thread.x)),
                   static_cast<unsigned long>(std::ceil(texDim.y / thread.y)));
        tracePath_kernel<<< block, thread >>>(surface,
                                              (float4 *)scaleViewInvEye,
                                              shapes,
                                              numShapes,
                                              areaLights,
                                              numAreaLights,
                                              texDim,
                                              randState);
    }
}
