#include <curand.h>
#include <curand_kernel.h>
#include "helper_cuda.h"


extern "C"
{
    /**
     * @brief setup_kernel
     * @param state
     * @param seed
     */
    __global__
    void kernel_initCuRand(curandState *state, uint64_t seed, dim3 texDim)
    {
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < texDim.x && y < texDim.y)
        {
            uint id = y * texDim.x + x;
            curand_init(seed, id, 0, &state[id]);
        }
    }


    /**
     * @brief cuda_LinitCuRand
     * @param state
     * @param seed
     */
    void cuda_initCuRand(curandState *state, uint64_t seed, dim3 texDim)
    {
        dim3 thread(32, 32);
        dim3 block(static_cast<unsigned long>(std::ceil(texDim.x / thread.x)),
                   static_cast<unsigned long>(std::ceil(texDim.y / thread.y)));
        kernel_initCuRand<<<block, thread>>>(state, seed, texDim);
    }

}
