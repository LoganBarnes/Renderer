#ifndef RENDER_OBJECTS_H
#define RENDER_OBJECTS_H

#include "helper_math.h"

/*
 * Matrix vector multiplication
 */
inline __host__ __device__ float4 operator*(float4 *mat4, float4 vec)
{
    return make_float4(dot(mat4[0], vec),
                       dot(mat4[1], vec),
                       dot(mat4[2], vec),
                       dot(mat4[3], vec));
}

/*
 * Object definitions
 */

enum ShapeType
{
    AABB, CONE, CUBE, CYLINDER, QUAD, SPHERE, TRIANGLE, NUM_SHAPE_TYPES
};

enum LightType
{
    POINT, DIRECTIONAL, AREA, SPOT, NUM_LIGHT_TYPES
};

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
    float invTrans[16];
    Material mat;
};

struct Light
{
    LightType type;
    float4 radiance;
};



#endif // RENDER_OBJECTS_H
