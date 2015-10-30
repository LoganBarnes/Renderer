#include <GL/glew.h>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include "pathTracer.hpp"
#include "RendererConfig.hpp"
#include "CudaFunctions.cuh"
#include "renderObjects.hpp"

//#include <glm/gtx/string_cast.hpp>
//#include <iostream>

PathTracer::PathTracer()
{
    cuda_init(0, NULL);
    cuda_allocateArray((void**)&m_dScaleViewInvEye, 16 * sizeof(float));
}


PathTracer::~PathTracer()
{
    cuda_freeArray(m_dScaleViewInvEye);
    cuda_destroy();
}


void PathTracer::register2DTexture(const char *name, GLuint tex)
{
    cudaGraphicsResource_t resource;
    cuda_registerGLTexture(&resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    m_resources[name] = resource;
}

void PathTracer::unregisterTexture(const char *name)
{
    cuda_unregisterResource(m_resources[name]);
}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
    glm::mat4 scaleViewInvT = glm::transpose(scaleViewInv);
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(scaleViewInvT), 0, 12*sizeof(float));
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(eye), 12*sizeof(float), 4*sizeof(float));
}


void PathTracer::tracePath(const char *tex, GLuint width, GLuint height)
{
    cudaGraphicsResource_t res = m_resources[tex];

    cuda_graphicsMapResource(&res);

    cudaArray_t writeArray;
    cuda_graphicsSubResourceGetMappedArray(&writeArray, res, 0, 0);

    cudaResourceDesc dsc;
    dsc.resType = cudaResourceTypeArray;
    dsc.res.array.array = writeArray;
    cudaSurfaceObject_t surface;
    cuda_createSurfaceObject(&surface, &dsc);

    cuda_tracePath(surface, m_dScaleViewInvEye, dim3(width, height));

    cuda_destroySurfaceObject(surface);
    cuda_graphicsUnmapResource(&res);
    cuda_streamSynchronize(0);
}


