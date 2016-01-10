#include <stdlib.h>
#include <time.h>
#include <glm/glm.hpp>
#include "renderObjects.hpp"


/**
 * @brief cuda_LinitCuRand
 * @param state
 * @param seed
 */
void cpu_initCuRand(uint64_t seed)
{
    if (seed)
        srand(seed);
    else
        srand(time(NULL));
}


/**
 * @brief randHemi
 * @param normal
 * @param state
 * @param id
 * @return
 */
float3 randHemi(float3 normal)
{
    float3 random;
    random.x = (rand() / RAND_MAX) * 1.999999f - 1.f;
    random.y = (rand() / RAND_MAX) * 1.999999f - 1.f;
    random.z = (rand() / RAND_MAX) * 1.999999f - 1.f;

    if (dot(normal, random) < 0.f)
        random = -random;

    return random;
}


/**
 * @brief kernel_randCosHemi
 * @param state
 * @param idx
 * @return
 */
float3 randCosHemi(float3 normal)
{
    const float e1 = 1.f - (rand() / RAND_MAX);
    const float e2 = 1.f - (rand() / RAND_MAX);

    // Jensen's method
    const float sin_theta = glm::sqrt(1.0f - e1);
    const float cos_theta = glm::sqrt(e1);
    const float phi = 6.28318531f * e2;

    float3 rand = glm::vec3(glm::cos(phi) * sin_theta,
                            glm::sin(phi) * sin_theta,
                              cos_theta);

    // Make a coordinate system
    const float3& Z = normal;

    float3 X, Y;

    // GET TANGENTS
    X = (glm::abs(normal.x) < 0.9f) ? glm::vec3(1.f, 0.f, 0.f) : glm::vec3(0.f, 1.f, 0.f);

    // Remove the part that is parallel to Z
    X -= normal * dot(normal, X);
    X /= length(X); // normalize no?
//        X = normalize(X);

    Y = cross(normal, X);

    return
        rand.x * X +
        rand.y * Y +
        rand.z * Z;
}


/**
 * @brief samplePoint
 * @param state
 * @param id
 * @param s
 * @return
 */
SurfaceElement samplePoint(Shape shape)
{
    SurfaceElement surfel;

    if (shape.type == QUAD)
    {
        float x = 1.f - (rand() / RAND_MAX);
        float y = 1.f - (rand() / RAND_MAX);

        x = x * 2.f - 1.f;
        y = y * 2.f - 1.f;

//            surfel.point = glm::vec3(shape.trans * make_float4(0.f, 0.f, 0.f, 1.f));
        surfel.point = glm::vec3(shape.trans * glm::vec4(x, y, 0.f, 1.f));
        surfel.normal = glm::normalize(shape.normInv * glm::vec3(0.f, 0.f, -1.f));
        surfel.material = shape.material;
        surfel.index = static_cast<int>(shape.index);
    }
    else
    {
        surfel.index = -1;
    }

    return surfel;
}

