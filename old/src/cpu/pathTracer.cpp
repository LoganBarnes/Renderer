#include <stdio.h>
#include <pthread.h>
#include <iostream>
#include <glm.hpp>
#include "renderObjects.hpp"
//#include "intersections.cu"
#include "renderRandom.cpp"

const float BUMP_VAL = 1e-3f;
const float PI_F = 3.141592653539f;


///**
// * @brief scatter
// * @param surfel
// * @param w_i
// * @param w_o
// * @param weight_o
// * @param eta_o
// * @param extinction_o
// * @param randState
// * @param id
// * @return
// */
//bool scatter(SurfaceElement *surfel,
//             const glm::vec3 &w_i,
//             glm::vec3 *w_o,
//             Radiance3 *weight_o,
//             float *eta_o,
//             curandState *randState,
//             int id)
//{
////        const glm::vec3 &n = surfel->normal;

//    float r = curand_uniform(randState + id);

//    Material &material = surfel->material;
//    if (dot(material.lambertianReflect, material.lambertianReflect) > 1e-7)
//    {
//        glm::vec3 &lambRefl = material.lambertianReflect;
//        float p_lambertianAvg = (lambRefl.x + lambRefl.y + lambRefl.z) / 3.f;
//        r -= p_lambertianAvg;

//        if (r < 0.f)
//        {
//            *weight_o = material.lambertianReflect / p_lambertianAvg;
//            *w_o = randCosHemi(surfel->normal, randState, id);
//            *eta_o = material.etaPos;

//            return true;
//        }
//    }

//    return false;
//}


///**
// * @brief estimateIndirectLight
// * @param surfel
// * @param ray
// * @param shapes
// * @param numShapes
// * @param randState
// * @param id
// * @return
// */
//glm::vec3 estimateIndirectLight(SurfaceElement *surfel,
//                                Ray *ray,
//                                curandState *randState,
//                                int id)
//{
//    glm::vec3 w_i = -ray->dir;
//    glm::vec3 w_o;
//    glm::vec3 coeff = glm::vec3(0.f);
//    float eta_o = 0.f;

//    if (scatter(surfel, w_i, &w_o, &coeff, &eta_o, randState, id))
//    {
//        float eta_i = surfel->material.etaPos;
//        float refractiveScale = eta_i / eta_o;
//        refractiveScale *= refractiveScale;

//        coeff *= refractiveScale;
//        float bump = BUMP_VAL;
//        if (dot(surfel->normal, w_o) < 0.f)
//        {
//            bump = -bump;
////                printf("neg bump ");
//        }
//        ray->orig = surfel->point + surfel->normal * bump;
//        ray->dir = w_o;
//    }
//    else
//        ray->isValid = false;

//    return coeff;
//}


///**
// * @brief estimateDirectLightFromAreaLights
// * @param surfel
// * @param ray
// * @param areaLights
// * @param numAreaLights
// * @return
// */
//glm::vec3 estimateDirectLightFromAreaLights(SurfaceElement *surfel,
//                                            Shape *shapes,
//                                            uint numShapes,
//                                            Shape *areaLights,
//                                            uint numAreaLights,
//                                            curandState *randState,
//                                            int id)
//{
//    Radiance3 L_o = glm::vec3(0.f);

//    for (uint l = 0; l < numAreaLights; ++l)
//    {
//        SurfaceElement lightSurfel = samplePoint(randState, id, areaLights[l]);

//        Ray r;
//        r.orig = surfel->point + surfel->normal * BUMP_VAL;
//        r.dir = normalize((lightSurfel.point + lightSurfel.normal * BUMP_VAL) - r.orig);
//        SurfaceElement intersection;
//        if (intersectWorld(&r, shapes, numShapes, &intersection, -1) &&
//                intersection.index == lightSurfel.index)
//        {
//            glm::vec3 w_i = lightSurfel.point - surfel->point;
//            const float distance2 = dot(w_i, w_i);
//            w_i /= sqrt(distance2);

//            L_o += surfel->material.color * // should calc BDSF
//                    (lightSurfel.material.power / PI_F) *
//                    max(0.f, dot(w_i, surfel->normal)) *
//                    max(0.f, dot(-w_i, lightSurfel.normal / distance2));
//        }
//        // TODO: implement impulses (specular)
//    }

//    return L_o;
//}


///**
// * @brief pathTrace
// * @param ray
// * @param coeff
// * @param shapes
// * @param numShapes
// * @param areaLights
// * @param numAreaLights
// * @param isEyeRay
// * @return
// */
//Radiance3 pathTrace(Ray *ray,
//                    glm::vec3 *coeff,
//                    Shape *shapes,
//                    uint numShapes,
//                    Shape *areaLights,
//                    uint numAreaLights,
//                    bool isEyeRay,
//                    curandState *randState,
//                    int id,
//                    bool debugEmit,
//                    bool debugDirect,
//                    bool debugIndirect)
//{
//    Radiance3 L_o = glm::vec3(0.f);

//    SurfaceElement surfel;
//    if (intersectWorld(ray, shapes, numShapes, &surfel, -1, isEyeRay))
//    {
//        if (isEyeRay && debugEmit)
//            L_o += *coeff * surfel.material.emitted;

//        if (!isEyeRay || debugDirect)
//        {
//            L_o += *coeff * estimateDirectLightFromAreaLights(&surfel,
//                                                             shapes,
//                                                             numShapes,
//                                                             areaLights,
//                                                             numAreaLights,
//                                                             randState,
//                                                             id);
//        } // end DIRECT

//        if (!isEyeRay || debugIndirect)
//        {
//            *coeff *= estimateIndirectLight(&surfel, ray, randState, id);
//            if (length(*coeff) < 1.e-9f)
//                ray->isValid = false;
//        }
//        if (!debugIndirect)
//            ray->isValid = false;
//    }
//    else
//        ray->isValid = false;

//    return L_o;
//}


///**
// * @brief tracePath_kernel
// * @param surfObj
// * @param scaleViewInvEye
// * @param shapes
// * @param numShapes
// * @param areaLights
// * @param numAreaLights
// * @param texDim
// * @param randState
// */
//void tracePath_kernel(cudaSurfaceObject_t surfObj,
//                      glm::vec4 *scaleViewInvEye,
//                      Shape *shapes,
//                      uint numShapes,
//                      Shape *areaLights,
//                      uint numAreaLights,
//                      glm::ivec3 texDim,
//                      curandState *randState,
//                      int bounceLimit,
//                      float scale,
//                      bool debugEmit,
//                      bool debugDirect,
//                      bool debugIndirect)
//{
//    // Calculate surface coordinates
//    uint x = blockIdx.x * blockDim.x + threadIdx.x;
//    uint y = blockIdx.y * blockDim.y + threadIdx.y;

//    if (x < texDim.x && y < texDim.y)
//    {
//        uint id = y * texDim.x + x;
//        glm::vec4 data;

//        // Read from input surface
//        surf2Dread(&data,  surfObj, x * sizeof(glm::vec4), y);

//        glm::vec2 coords = glm::vec2((x + 0.5f) / texDim.x, (y + 0.5f) / texDim.y);
//        coords = (coords * 2.f) - 1.f; // screen space

//        glm::vec4 farPoint = glm::vec4(coords.x, coords.y, -1, 1);

//        Ray ray;
//        ray.orig = glm::vec3(scaleViewInvEye[4]);
//        ray.dir = glm::vec3(scaleViewInvEye * farPoint);
//        ray.dir = normalize(ray.dir - ray.orig);
//        ray.isValid = true;

//        glm::vec3 coeff = glm::vec3(1.f);
//        Radiance3 radiance = glm::vec3(0.f);

//        radiance += pathTrace(&ray,
//                              &coeff,
//                              shapes,
//                              numShapes,
//                              areaLights,
//                              numAreaLights,
//                              true,
//                              randState,
//                              id,
//                              debugEmit,
//                              debugDirect,
//                              debugIndirect);

//        int iteration = 1;
//        while (ray.isValid && iteration++ < bounceLimit)
//        {
//            radiance += pathTrace(&ray,
//                                  &coeff,
//                                  shapes,
//                                  numShapes,
//                                  areaLights,
//                                  numAreaLights,
//                                  false,
//                                  randState,
//                                  id,
//                                  debugEmit,
//                                  debugDirect,
//                                  debugIndirect);
//        }

//        glm::vec4 result = glm::vec4(radiance * scale, 1.f);

//        // Write to output surface
//        surf2Dwrite(result, surfObj, x * sizeof(glm::vec4), y);

//    }
//}




void *asyncPathTrace(void *args)
{
    ThreadData *threadData = reinterpret_cast<ThreadData*>(args);

    float *data = threadData->data;
    for (uint r = threadData->rowStart; r < threadData->rowEnd; ++r)
    {
        for (uint c = 0; c < threadData->colWidth; ++c)
        {
            uint i = r * threadData->colWidth + c;
            data[i*4  ] = threadData->alpha;
            data[i*4+1] = 1.f;
            data[i*4+2] = threadData->alpha;
            data[i*4+3] = 1.0f;
        }
    }

    if (threadData->isMainThread)
        return NULL;

    pthread_exit(NULL);
}


/**
 * @brief cpu_tracePath
 * @param surface
 * @param scaleViewInvEye
 * @param shapes
 * @param numShapes
 * @param areaLights
 * @param numAreaLights
 * @param texDim
 * @param randState
 */
//void cpu_tracePath(float *data,
//                   glm::mat4 scaleViewInv,
//                   glm::vec4 eye,
//                   Shape *shapes,
//                   uint numShapes,
//                   Shape *areaLights,
//                   uint numAreaLights,
//                   glm::ivec3 texDim,
//                   uint numThreads,
//                   pthread_t *threads,
//                   pthread_attr_t attr,
//                   bool debugEmit,
//                   bool debugDirect,
//                   bool debugIndirect,
//                   int bounceLimit = 1000,
//                   float scale = 1.f)
void cpu_tracePath(float *data,
                   glm::mat4 ,
                   glm::vec4 ,
                   Shape *,
                   uint ,
                   Shape *,
                   uint ,
                   glm::ivec3 texDim,
                   uint numThreads,
                   pthread_t *threads,
                   pthread_attr_t attr,
                   ThreadData *args,
                   bool ,
                   bool ,
                   bool ,
                   int ,
                   float scale)
{


//    tracePath_kernel(static_cast<glm::vec4*>(data),
//                     scaleViewInv,
//                     eye,
//                     shapes,
//                     numShapes,
//                     areaLights,
//                     numAreaLights,
//                     texDim,
//                     bounceLimit,
//                     scale,
//                     debugEmit,
//                     debugDirect,
//                     debugIndirect);

    uint width = static_cast<uint>(texDim.x);
    uint height = static_cast<uint>(texDim.y);
    uint div = height / numThreads;
    uint rowEnd = 0;

    int rc;

//    uint s = 1;
//    rowEnd = s * div;

    for (uint i = 0; i < numThreads - 1; ++i)
//    for (uint i = s; i < s+1; ++i)
    {
        ThreadData &threadData = args[i];
        threadData.data = data;
        threadData.alpha = 1.f * scale;

        threadData.rowStart = rowEnd;
        rowEnd = (i+1) * div;
        threadData.rowEnd = rowEnd;

        threadData.colWidth = width;
        threadData.isMainThread = false;
        threadData.alpha = rowEnd * 1.f / height;
        rc = pthread_create(&threads[i], &attr, asyncPathTrace, reinterpret_cast<void*>(&threadData));
        if (rc){
           std::cout << "ERROR: failed creating thread, " << rc << std::endl;
        }
    }
    ThreadData &threadData = args[numThreads-1];
    threadData.data = data;
    threadData.alpha = 1.f * scale;

    threadData.rowStart = rowEnd;
    rowEnd = height;
    threadData.rowEnd = rowEnd;

    threadData.colWidth = width;
    threadData.isMainThread = true;
    threadData.alpha = rowEnd * 1.f / height;
    asyncPathTrace(reinterpret_cast<void*>(&threadData));

    void *status;
    for (uint i = 0; i < numThreads - 1; ++i)
//    for (uint i = s; i < s+1; ++i)
    {
        rc = pthread_join(threads[i], &status);
        if (rc){
           std::cout << "ERROR: unable to join," << rc << std::endl;
        }
    }

}

