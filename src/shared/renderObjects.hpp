#ifndef RENDER_OBJECTS_H
#define RENDER_OBJECTS_H

#include <glm/glm.hpp>
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
 * Object definitions
 */

struct Ray
{
    float3 orig;
    float3 dir;
};

struct Material
{
    float4 color;
};

struct Shape
{
    ShapeType type;
    float inv[16];
    Material mat;
};

struct Luminaire
{
    LuminaireType type;
    float4 radiance;
};

#else

/*
 * Object definitions
 */

struct Ray
{
    glm::vec3 orig;
    glm::vec3 dir;
};

struct Material
{
    glm::vec4 color;
};

struct Shape
{
    ShapeType type;
    glm::mat4 inv;
    Material mat;
};

struct Luminaire
{
    LuminaireType type;
    glm::mat4 radiance;
};

#endif // USE_CUDA


#endif // RENDER_OBJECTS_H
