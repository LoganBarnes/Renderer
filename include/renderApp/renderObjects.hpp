#ifndef RENDER_OBJECTS_H
#define RENDER_OBJECTS_H

#include <glm.hpp>
#include "renderTypes.hpp"
#include "renderer-config.hpp"

#ifdef USE_CUDA
#include "helper_math.h"


/*
 * Matrix initializer. Matrix stored in column major order.
 */
inline __host__ __device__ void set_float_mat4(float4 *mat4, glm::mat4 otherMat)
{
    for (int c = 0; c < 4; ++c)
    {
        mat4[c] = make_float4(otherMat[c][0], otherMat[c][1], otherMat[c][2], otherMat[c][3]);
    }
}


/*
 * Matrix initializer. Matrix stored in column major order.
 */
inline __host__ __device__ void set_float_mat3(float3 *mat3, glm::mat3 otherMat)
{
    for (int c = 0; c < 3; ++c)
    {
        mat3[c] = make_float3(otherMat[c][0], otherMat[c][1], otherMat[c][2]);
    }
}


/*
 * Matrix initializer. Matrix stored in column major order.
 */
inline __host__ __device__ void make_float_mat3(float3 *mat3, float4 *mat4)
{
    mat3[0] = make_float3(mat4[0]);
    mat3[1] = make_float3(mat4[1]);
    mat3[2] = make_float3(mat4[2]);
}


/*
 * Matrix vector multiplication. Matrix stored in column major order.
 */
inline __host__ __device__ float4 operator*(float4 *mat4, float4 vec)
{
    //    return make_float4(dot(mat4[0], vec),
    //                       dot(mat4[1], vec),
    //                       dot(mat4[2], vec),
    //                       dot(mat4[3], vec));
    return mat4[0] * vec.x + mat4[1] * vec.y + mat4[2] * vec.z + mat4[3] * vec.w;
}


/*
 * Matrix vector multiplication. Matrix stored in column major order.
 */
inline __host__ __device__ float3 operator*(float3 *mat3, float3 vec)
{
    //    return make_float3(dot(mat3[0], vec),
    //                       dot(mat3[1], vec),
    //                       dot(mat3[2], vec));
    return mat3[0] * vec.x + mat3[1] * vec.y + mat3[2] * vec.z;
}


/*
 * Transpose Matrix.
 */
inline __host__ __device__ void transpose_float_mat3(float3 *out, float3 *in)
{
    out[0] = make_float3(in[0].x, in[1].x, in[2].x);
    out[1] = make_float3(in[0].y, in[1].y, in[2].y);
    out[2] = make_float3(in[0].z, in[1].z, in[2].z);
}

#else

typedef glm::vec3 float3;
typedef unsigned int uint;

#endif // USE_CUDA

/*
 * Object definitions
 */

typedef float3 Radiance3;

struct Ray
{
    float3 orig;
    float3 dir;
    bool isValid;
};

struct Material
{
    float3 color;
    float3 power;
    Radiance3 emitted;
    float3 lambertianReflect;
    float etaPos; // refractive index out of surface
    float etaNeg; // refractive index in surface
};

struct SurfaceElement
{
    float3 point;
    float3 normal;
    Material material;
    int index;
};

struct PathChoice
{
    Radiance3 radiance;
    float coeff;
    Ray scatter;
};

#ifdef USE_CUDA

struct Shape
{
    ShapeType type;
    float4 trans[4];
    float4 inv[4];
    float3 normInv[3];
    Material material;
    uint index;
};

#else

struct Shape
{
    ShapeType type;
    glm::mat4 trans;
    glm::mat4 inv;
    glm::mat3 normInv;
    Material material;
    uint index;
};

struct ThreadData
{
    float *data;
    uint rowStart;
    uint rowEnd;
    uint colWidth;
    float alpha;
    bool isMainThread;
};

#endif


#endif // RENDER_OBJECTS_H
