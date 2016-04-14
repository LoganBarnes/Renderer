#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "helper_grid.h"
#include "renderObjects.hpp"
#include "intersections.cu"
#include "renderRandom.cu"

__device__ const float BUMP_VAL = 1e-3f;
__device__ const float PI_F = 3.141592653539f;

extern "C"
{

    /**
     * @brief scatter
     * @param surfel
     * @param w_i
     * @param w_o
     * @param weight_o
     * @param eta_o
     * @param extinction_o
     * @param randState
     * @param id
     * @return
     */
    __device__
    bool scatter(SurfaceElement *surfel,
                 const float3 &w_i,
                 float3 *w_o,
                 Radiance3 *weight_o,
                 float *eta_o,
                 curandState *randState,
                 int id)
    {
//        const float3 &n = surfel->normal;

        float r = curand_uniform(randState + id);

        Material &material = surfel->material;
        if (dot(material.lambertianReflect, material.lambertianReflect) > 1e-7)
        {
            float3 &lambRefl = material.lambertianReflect;
            float p_lambertianAvg = (lambRefl.x + lambRefl.y + lambRefl.z) / 3.f;
            r -= p_lambertianAvg;

            if (r < 0.f)
            {
                *weight_o = material.lambertianReflect / p_lambertianAvg;
                *w_o = randCosHemi(surfel->normal, randState, id);
                *eta_o = material.etaPos;

                return true;
            }
        }

        return false;
    }

    /**
     * @brief estimateIndirectLight
     * @param surfel
     * @param ray
     * @param shapes
     * @param numShapes
     * @param randState
     * @param id
     * @return
     */
    __device__
    float3 estimateIndirectLight(SurfaceElement *surfel,
                                    Ray *ray,
                                    curandState *randState,
                                    int id)
    {
        float3 w_i = -ray->dir;
        float3 w_o;
        float3 coeff = make_float3(0.f);
        float eta_o = 0.f;

        if (scatter(surfel, w_i, &w_o, &coeff, &eta_o, randState, id))
        {
            float eta_i = surfel->material.etaPos;
            float refractiveScale = eta_i / eta_o;
            refractiveScale *= refractiveScale;

            coeff *= refractiveScale;
            float bump = BUMP_VAL;
            if (dot(surfel->normal, w_o) < 0.f)
            {
                bump = -bump;
//                printf("neg bump ");
            }
            ray->orig = surfel->point + surfel->normal * bump;
            ray->dir = w_o;
        }
        else
            ray->isValid = false;

        return coeff;
    }


    /**
     * @brief estimateDirectLightFromAreaLights
     * @param surfel
     * @param ray
     * @param areaLights
     * @param numAreaLights
     * @return
     */
    __device__
    float3 estimateDirectLightFromAreaLights(SurfaceElement *surfel,
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
            r.orig = surfel->point + surfel->normal * BUMP_VAL;
            r.dir = normalize((lightSurfel.point + lightSurfel.normal * BUMP_VAL) - r.orig);
            SurfaceElement intersection;
            if (intersectWorld(&r, shapes, numShapes, &intersection, -1) &&
                    intersection.index == lightSurfel.index)
            {
                float3 w_i = lightSurfel.point - surfel->point;
                const float distance2 = dot(w_i, w_i);
                w_i /= sqrt(distance2);

                L_o += surfel->material.color * // should calc BSDF
                        (lightSurfel.material.power / PI_F) *
                        max(0.f, dot(w_i, surfel->normal)) *
                        max(0.f, dot(-w_i, lightSurfel.normal / distance2));
            }
            // TODO: implement impulses (specular)
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
    Radiance3 pathTrace(Ray *ray,
                        float3 *coeff,
                        Shape *shapes,
                        uint numShapes,
                        Shape *areaLights,
                        uint numAreaLights,
                        bool isEyeRay,
                        curandState *randState,
                        int id,
                        bool debugEmit,
                        bool debugDirect,
                        bool debugIndirect)
    {
        Radiance3 L_o = make_float3(0.f);

        SurfaceElement surfel;
        if (intersectWorld(ray, shapes, numShapes, &surfel, -1, isEyeRay))
        {
            if (isEyeRay && debugEmit)
                L_o += *coeff * surfel.material.emitted;

            if (!isEyeRay || debugDirect)
            {
                L_o += *coeff * estimateDirectLightFromAreaLights(&surfel,
                                                                 shapes,
                                                                 numShapes,
                                                                 areaLights,
                                                                 numAreaLights,
                                                                 randState,
                                                                 id);
            } // end DIRECT

            if (!isEyeRay || debugIndirect)
            {
                *coeff *= estimateIndirectLight(&surfel, ray, randState, id);
                if (length(*coeff) < 1.e-9f)
                    ray->isValid = false;
            }
            if (!debugIndirect)
                ray->isValid = false;
        }
        else
            ray->isValid = false;

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
                          curandState *randState,
                          int bounceLimit,
                          float scale,
                          bool debugEmit,
                          bool debugDirect,
                          bool debugIndirect)
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

            float3 coeff = make_float3(1.f);
            Radiance3 radiance = make_float3(0.f);

            radiance += pathTrace(&ray,
                                  &coeff,
                                  shapes,
                                  numShapes,
                                  areaLights,
                                  numAreaLights,
                                  true,
                                  randState,
                                  id,
                                  debugEmit,
                                  debugDirect,
                                  debugIndirect);

            int iteration = 1;
            while (ray.isValid && iteration++ < bounceLimit)
            {
                radiance += pathTrace(&ray,
                                      &coeff,
                                      shapes,
                                      numShapes,
                                      areaLights,
                                      numAreaLights,
                                      false,
                                      randState,
                                      id,
                                      debugEmit,
                                      debugDirect,
                                      debugIndirect);
            }

            float4 result = make_float4(radiance * scale, 1.f);

//            // nans?
//            result.x = result.x != result.x ? 0.f : result.x;
//            result.y = result.y != result.y ? 0.f : result.y;
//            result.z = result.z != result.z ? 0.f : result.z;

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
                        curandState *randState,
                        bool debugEmit,
                        bool debugDirect,
                        bool debugIndirect,
                        int bounceLimit = 1000,
                        float scale = 1.f)
    {
        dim3 thread(32, 32);
        dim3 block(1);
        computeGridSize(texDim.x, thread.x, block.x, thread.x);
        computeGridSize(texDim.y, thread.y, block.y, thread.y);

        tracePath_kernel<<< block, thread >>>(surface,
                                              (float4 *)scaleViewInvEye,
                                              shapes,
                                              numShapes,
                                              areaLights,
                                              numAreaLights,
                                              texDim,
                                              randState,
                                              bounceLimit,
                                              scale,
                                              debugEmit,
                                              debugDirect,
                                              debugIndirect);
    }
}
