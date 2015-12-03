#include <curand.h>
#include <curand_kernel.h>
#include "renderObjects.hpp"


extern "C"
{

    /**
     * @brief kernel_randCosHemi
     * @param state
     * @param idx
     * @return
     */
    __device__
    float3 randCosHemi(curandState *state, int id)
    {
        float theta = 1.f - curand_uniform(state + id);
        float s = 1.f - curand_uniform(state + id);
        float y = sqrt(s);
        float r = sqrt(1.f - y * y);
        return make_float3(r * cos(theta), y, r * sin(theta));
    }


    /**
     * @brief samplePoint
     * @param state
     * @param id
     * @param s
     * @return
     */
    __device__
    SurfaceElement samplePoint(curandState *state, int id, Shape shape)
    {
        SurfaceElement surfel;

        if (shape.type == QUAD)
        {
            float x = 1.f - curand_uniform(state + id);
            float y = 1.f - curand_uniform(state + id);

            x = x * 2.f - 1.f;
            y = y * 2.f - 1.f;

//            surfel.point = make_float3(shape.trans * make_float4(0.f, 0.f, 0.f, 1.f));
            surfel.point = make_float3(shape.trans * make_float4(x, y, 0.f, 1.f));
            surfel.normal = shape.normInv * make_float3(0.f, 0.f, -1.f);
            surfel.material = shape.material;
            surfel.index = static_cast<int>(shape.index);
        }
        else
        {
            surfel.index = -1;
        }

        return surfel;
    }


}
