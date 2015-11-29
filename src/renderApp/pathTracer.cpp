#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include "pathTracer.hpp"
#include "renderObjects.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "CudaFunctions.cuh"
#endif

PathTracer::PathTracer()
#ifdef USE_CUDA
    : m_dScaleViewInvEye(NULL),
      m_dShapes(NULL),
      m_dLuminaires(NULL),
#else
    : m_hShapes(NULL),
      m_hLuminaires(NULL),
#endif
      m_numShapes(0),
      m_numLuminaires(0),
      m_initialized(false)
{}


PathTracer::~PathTracer()
{
#ifdef USE_CUDA
    if (m_dScaleViewInvEye)
    {
        cuda_freeArray(m_dScaleViewInvEye);
        m_dScaleViewInvEye = NULL;
    }
    if (m_dShapes)
    {
        cuda_freeArray(m_dShapes);
        m_dShapes = NULL;
    }
    if (m_dLuminaires)
    {
        cuda_freeArray(m_dLuminaires);
        m_dShapes = NULL;
    }

    if (m_initialized)
        cuda_destroy();
#else
    if (m_hShapes)
    {
        delete[] m_hShapes;
        m_hShapes = NULL;
    }
    if (m_hLuminaires)
    {
        delete[] m_hLuminaires;
        m_hShapes = NULL;
    }
#endif
}


#ifdef USE_CUDA
void PathTracer::init(int argc, const char **argv)
{
    cuda_init(argc, argv);
    m_initialized = true;

    cuda_allocateArray(reinterpret_cast<void**>(&m_dScaleViewInvEye), 20 * sizeof(float));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dShapes), MAX_DEVICE_SHAPES * sizeof(Shape));
#else
void PathTracer::init(int, const char**)
{
    m_initialized = true;
    m_hShapes = new Shape[MAX_DEVICE_SHAPES];
#endif
}


void PathTracer::register2DTexture(const char *name, GLuint tex)
{
#ifdef USE_CUDA
    cudaGraphicsResource_t resource;
    cuda_registerGLTexture(&resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    m_resources[name] = resource;
#else
    m_textures[name] = tex;
#endif
}

void PathTracer::unregisterTexture(const char *name)
{
#ifdef USE_CUDA
    cuda_unregisterResource(m_resources[name]);
#else
    m_textures.count(name);
#endif
}


void PathTracer::addShape(ShapeType type, glm::mat4 trans, glm::vec4 color)
{
    Shape shape;
    shape.type = type;
#ifdef USE_CUDA
    glm::mat4 inv = glm::inverse(trans);
    set_float_mat4(reinterpret_cast<float4*>(shape.inv), inv);
    shape.mat.color = make_float4(color.x, color.y, color.z, color.w);

    cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));
#else
    shape.inv = glm::inverse(trans);
    shape.mat.color = color;
    m_hShapes[m_numShapes] = shape;
#endif
    ++m_numShapes;
}


void PathTracer::addLuminaire()
{

}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
//    glm::mat4 scaleViewInvT = glm::transpose(scaleViewInv);
#ifdef USE_CUDA
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(scaleViewInv), 0, 16*sizeof(float));
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(eye), 16*sizeof(float), 4*sizeof(float));
#else
    m_hScaleViewInv = scaleViewInv;
    m_hEye = eye;
#endif
}


void PathTracer::tracePath(const char *writeTex, GLuint width, GLuint height)
{
#ifdef USE_CUDA
    this->_tracePathCUDA(writeTex, width, height);
#else
    this->_tracePathCPU(writeTex, width, height);
#endif
}

#ifdef USE_CUDA
void PathTracer::_tracePathCUDA(const char *tex, GLuint width, GLuint height)
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

#else
void PathTracer::_tracePathCPU(const char *writeTex, GLuint width, GLuint height)
{
    GLsizei widthi = static_cast<GLsizei>(width);
    GLsizei heighti = static_cast<GLsizei>(height);
    GLsizei numPix = widthi * heighti;
    GLsizei size = numPix * 4;
    float *data = new float[size];

    for (GLsizei i = 0; i < numPix; ++i)
    {
        data[i*4  ] = 0.f;
        data[i*4+1] = 1.f;
        data[i*4+2] = 0.5f;
        data[i*4+3] = 1.f;
    }

    m_textures[writeTex];

#ifdef USE_GRAPHICS
    glBindTexture(GL_TEXTURE_2D, m_textures[writeTex]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, widthi, heighti, 0, GL_RGBA, GL_FLOAT, data);
#endif
    delete[] data;
}
#endif




