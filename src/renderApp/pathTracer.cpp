#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include "pathTracer.hpp"
#include "renderObjects.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "CudaFunctions.cuh"
#endif

const uint64_t RAND_SEED = 1337;

PathTracer::PathTracer()
#ifdef USE_CUDA
    : m_dScaleViewInvEye(NULL),
      m_dShapes(NULL),
      m_dAreaLights(NULL),
      m_dRandState(NULL),
#else
    : m_hShapes(NULL),
      m_hLuminaires(NULL),
#endif
      m_numShapes(0),
      m_numAreaLights(0),
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
    if (m_dAreaLights)
    {
        cuda_freeArray(m_dAreaLights);
        m_dShapes = NULL;
    }
    if (m_dRandState)
    {
        cuda_freeArray(m_dRandState);
        m_dRandState = NULL;
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
void PathTracer::init(int argc, const char **argv, GLuint width, GLuint height)
{
    cuda_init(argc, argv);
    m_initialized = true;

    cuda_allocateArray(reinterpret_cast<void**>(&m_dRandState), width * height * sizeof(curandState));
    cuda_initCuRand(m_dRandState, RAND_SEED, dim3(width, height));

    cuda_allocateArray(reinterpret_cast<void**>(&m_dScaleViewInvEye), 20 * sizeof(float));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dShapes), MAX_DEVICE_SHAPES * sizeof(Shape));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dAreaLights), MAX_DEVICE_AREA_LIGHTS * sizeof(Shape));
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


void PathTracer::addShape(ShapeType type, glm::mat4 trans, glm::vec3 color)
{
    if (m_numShapes >= MAX_DEVICE_SHAPES)
        return;

    Shape shape;
    shape.type = type;
#ifdef USE_CUDA
    glm::mat4 inv = glm::inverse(trans);
    glm::mat3 normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    set_float_mat4(shape.trans, trans);
    set_float_mat4(shape.inv, inv);
    set_float_mat3(shape.normInv, normInv);
    shape.material.color = make_float3(color.x, color.y, color.z);
    shape.material.power = make_float3(0.f);
    shape.material.emitted = make_float3(0.f);
    shape.index = m_numShapes;

    cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));
#else
    shape.inv = glm::inverse(trans);
    shape.mat.color = color;
    m_hShapes[m_numShapes] = shape;
#endif
    ++m_numShapes;
}


void PathTracer::addAreaLight(ShapeType type, glm::mat4 trans, glm::vec3 power, float area)
{
    if (m_numShapes >= MAX_DEVICE_SHAPES ||
            m_numAreaLights >= MAX_DEVICE_AREA_LIGHTS)
        return;

    Shape shape;
    shape.type = type;
#ifdef USE_CUDA
    glm::mat4 inv = glm::inverse(trans);
    glm::mat3 normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    set_float_mat4(shape.trans, trans);
    set_float_mat4(shape.inv, inv);
    set_float_mat3(shape.normInv, normInv);
    shape.material.color = make_float3(0.f);
    shape.material.power = make_float3(power.x, power.y, power.z);
    shape.material.emitted = shape.material.power / (M_PI * area);
    shape.index = m_numShapes;

    cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));
    cuda_copyArrayToDevice(m_dAreaLights, &shape, m_numAreaLights * sizeof(Shape), sizeof(Shape));
#else
    shape.inv = glm::inverse(trans);
    shape.mat.color = color;
    m_hShapes[m_numShapes] = shape;
#endif
    ++m_numShapes;
    ++m_numAreaLights;

}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
//    glm::mat4 scaleViewInvT = glm::transpose(scaleViewInv);
#ifdef USE_CUDA
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(scaleViewInv), 0, 16 * sizeof(float));
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(eye), 16*sizeof(float), 4 * sizeof(float));
#else
    m_hScaleViewInv = scaleViewInv;
    m_hEye = eye;
#endif
}


void PathTracer::tracePath(const char *writeTex, GLuint width, GLuint height, GLuint iteration)
{
#ifdef USE_CUDA
    this->_tracePathCUDA(writeTex, width, height, iteration);
#else
    this->_tracePathCPU(writeTex, width, height);
#endif
}

#ifdef USE_CUDA
void PathTracer::_tracePathCUDA(const char *tex, GLuint width, GLuint height, GLuint iteration)
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
                   m_dAreaLights,
                   m_numAreaLights,
                   dim3(width, height),
                   m_dRandState,
                   iteration);

    cuda_destroySurfaceObject(surface);
    cuda_graphicsUnmapResource(&res);
    cuda_streamSynchronize(0);
    cuda_deviceSynchronize();
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




