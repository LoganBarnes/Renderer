#include <GL/glew.h>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include "pathTracer.hpp"
#include "RendererConfig.hpp"
#include "CudaFunctions.cuh"
#include "renderObjects.hpp"

PathTracer::PathTracer()
    : m_numShapes(0),
      m_numLuminaires(0)
{
    cuda_init(0, NULL);

    cuda_allocateArray(reinterpret_cast<void**>(&m_dScaleViewInvEye), 20 * sizeof(float));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dShapes), MAX_DEVICE_SHAPES * sizeof(Shape));
}


PathTracer::~PathTracer()
{
    cuda_freeArray(m_dScaleViewInvEye);
    cuda_freeArray(m_dShapes);

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


void PathTracer::addShape(ShapeType type, glm::mat4 trans, glm::vec4 color)
{
    Shape shape;
    shape.type = type;
    glm::mat4 inv = glm::inverse(trans);
    set_float_mat4(reinterpret_cast<float4*>(shape.inv), inv);
    shape.mat.color = make_float4(color.x, color.y, color.z, color.w);

    cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));
    ++m_numShapes;
}


void PathTracer::addLuminaire()
{

}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
//    glm::mat4 scaleViewInvT = glm::transpose(scaleViewInv);
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(scaleViewInv), 0, 16*sizeof(float));
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(eye), 16*sizeof(float), 4*sizeof(float));
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

    cuda_tracePath(surface,
                   m_dScaleViewInvEye,
                   m_dShapes,
                   m_numShapes,
                   m_dLuminaires,
                   m_numLuminaires,
                   dim3(width, height));

    cuda_destroySurfaceObject(surface);
    cuda_graphicsUnmapResource(&res);
    cuda_streamSynchronize(0);
}


