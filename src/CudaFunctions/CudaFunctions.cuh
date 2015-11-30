#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>

typedef uint32_t GLenum;
typedef uint32_t GLuint;
typedef unsigned int uint;

struct Shape;

extern "C"
{
    /*
     * from 'wrappers.cu'
     */
    void cuda_init(int argc, const char **argv);
    void cuda_destroy();

    void cuda_allocateArray(void **devPtr, int size);
    void cuda_freeArray(void *devPtr);

    void cuda_copyArrayToDevice(void *device, const void *host, uint offset, uint size);
    void cuda_copyArrayFromDevice(void *host, const void *device, uint size);

    void cuda_registerGLTexture(cudaGraphicsResource_t* resource, GLuint tex, GLenum target, cudaGraphicsRegisterFlags flags);
    void cuda_unregisterResource(cudaGraphicsResource_t resource);

    void cuda_graphicsMapResource(cudaGraphicsResource_t *res);
    void cuda_graphicsUnmapResource(cudaGraphicsResource_t *res);

    void cuda_graphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t res, GLuint index, GLuint level);

    void cuda_createSurfaceObject(cudaSurfaceObject_t *surface, cudaResourceDesc *desc);
    void cuda_destroySurfaceObject(cudaSurfaceObject_t surface);

    void cuda_streamSynchronize(cudaStream_t stream);
    void cuda_deviceSynchronize();

    /*
     * from 'pathTracer.cu'
     */
    void cuda_tracePath(cudaSurfaceObject_t surface,
                        float *scaleViewInv,
                        Shape *shapes,
                        uint numShapes,
                        Shape *areaLights,
                        uint numAreaLights,
                        dim3 texDim,
                        curandState *randState);

    /*
     * from 'random.cu'
     */
    void cuda_initCuRand(curandState *state, uint64_t seed, dim3 texDim);

}

#endif // CUDA_FUNCTIONS_CUH
