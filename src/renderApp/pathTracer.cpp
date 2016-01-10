#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "pathTracer.hpp"
#include "renderObjects.hpp"

const bool EMIT = true;
const bool DIRECT = true;
const bool INDIRECT = true;

const int BOUNCE_LIMIT = 5;
const uint64_t RAND_SEED = 1337;


#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "cuda_wrappers.cuh"
#include "cuda_random.cuh"
#include "cuda_render.cuh"

#define USE_CUDA_PROFILING

PathTracer::PathTracer()
    : m_dScaleViewInvEye(NULL),
      m_dShapes(NULL),
      m_dAreaLights(NULL),
      m_dRandState(NULL),
      m_numShapes(0),
      m_numAreaLights(0),
      m_initialized(false)
{}


PathTracer::~PathTracer()
{
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

    if (m_hShapes)
    {
        delete[] m_hShapes;
        m_hShapes = NULL;
    }
    if (m_hAreaLights)
    {
        delete[] m_hAreaLights;
        m_hShapes = NULL;
    }
}


void PathTracer::init(int argc, const char **argv, GLuint width, GLuint height)
{
    cuda_init(argc, argv);
    m_initialized = true;

    cuda_allocateArray(reinterpret_cast<void**>(&m_dRandState), width * height * sizeof(curandState));

    std::cout << "Initializing random states..." << std::endl;
    // break up initialization to avoid timeout errors
    GLuint partsMinus = height / 200;
    GLuint parts = partsMinus + 1;
    GLuint initHeight = height / parts;
    GLuint offset;
    for (GLuint i = 0; i < partsMinus; ++i)
    {
        offset = i * initHeight * width;
        cuda_initCuRand(m_dRandState, offset, RAND_SEED, dim3(width, initHeight));
        std::cout << "\r(" << (i+1) << "/" << parts << ")";
    }
    offset = partsMinus * initHeight * width;
    GLuint finalHeight = height - (initHeight * partsMinus);
    cuda_initCuRand(m_dRandState, offset, RAND_SEED, dim3(width, finalHeight));
    std::cout << "\rdone   " << std::endl;

    cuda_allocateArray(reinterpret_cast<void**>(&m_dScaleViewInvEye), 20 * sizeof(float));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dShapes), MAX_SHAPES * sizeof(Shape));
    cuda_allocateArray(reinterpret_cast<void**>(&m_dAreaLights), MAX_AREA_LIGHTS * sizeof(Shape));

    // host
    m_hShapes = new Shape[MAX_SHAPES];
    m_hAreaLights = new Shape[MAX_AREA_LIGHTS];
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


void PathTracer::swapResources(const char *res1, const char *res2)
{
    cudaGraphicsResource_t temp = m_resources[res1];
    m_resources[res1] = m_resources[res2];
    m_resources[res2] = temp;
}


void PathTracer::addShape(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU)
{
    if (m_numShapes >= MAX_SHAPES)
        return;

    Shape shape;
    shape.type = type;
    glm::mat4 inv = glm::inverse(trans);
    glm::mat3 normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    set_float_mat4(shape.trans, trans);
    set_float_mat4(shape.inv, inv);
    set_float_mat3(shape.normInv, normInv);
    shape.material = material;
    shape.index = m_numShapes;

    m_hShapes[m_numShapes] = shape;

    if (sendToGPU)
        cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));

    ++m_numShapes;
}


void PathTracer::addAreaLight(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU)
{
    if (m_numShapes >= MAX_SHAPES ||
            m_numAreaLights >= MAX_AREA_LIGHTS)
        return;

    Shape shape;
    shape.type = type;
    glm::mat4 inv = glm::inverse(trans);
    glm::mat3 normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    set_float_mat4(shape.trans, trans);
    set_float_mat4(shape.inv, inv);
    set_float_mat3(shape.normInv, normInv);
    shape.material = material;
    shape.index = m_numShapes;

    m_hShapes[m_numShapes] = shape;
    m_hAreaLights[m_numAreaLights] = shape;

    if (sendToGPU)
    {
        cuda_copyArrayToDevice(m_dShapes, &shape, m_numShapes * sizeof(Shape), sizeof(Shape));
        cuda_copyArrayToDevice(m_dAreaLights, &shape, m_numAreaLights * sizeof(Shape), sizeof(Shape));
    }

    ++m_numShapes;
    ++m_numAreaLights;
}


void PathTracer::updateShapesOnGPU()
{
    cuda_copyArrayToDevice(m_dShapes, m_hShapes, 0, m_numShapes * sizeof(Shape));
    cuda_copyArrayToDevice(m_dAreaLights, m_hAreaLights, 0, m_numAreaLights * sizeof(Shape));
}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(scaleViewInv), 0, 16 * sizeof(float));
    cuda_copyArrayToDevice(m_dScaleViewInvEye, glm::value_ptr(eye), 16*sizeof(float), 4 * sizeof(float));
}


void PathTracer::tracePath(const char *writeTex, GLuint width, GLuint height, float scaleFactor)
{
    this->_tracePathCUDA(writeTex, width, height, scaleFactor);
}


void PathTracer::_tracePathCUDA(const char *tex, GLuint width, GLuint height, float scaleFactor)
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

#ifdef USE_CUDA_PROFILING
    cuda_profilerStart();
#endif
    cuda_tracePath(surface,
                   m_dScaleViewInvEye,
                   m_dShapes,
                   m_numShapes,
                   m_dAreaLights,
                   m_numAreaLights,
                   dim3(width, height),
                   m_dRandState,
                   EMIT,
                   DIRECT,
                   INDIRECT,
                   BOUNCE_LIMIT,
                   scaleFactor);
#ifdef USE_CUDA_PROFILING
    cuda_profilerStop();
#endif

    cuda_destroySurfaceObject(surface);
    cuda_graphicsUnmapResource(&res);
    cuda_streamSynchronize(0);
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

#else

#include <unistd.h>

PathTracer::PathTracer()
    : m_hShapes(NULL),
      m_hAreaLights(NULL),
      m_threads(NULL),
      m_numShapes(0),
      m_numAreaLights(0),
      m_initialized(false)
{
    m_numThreads = sysconf( _SC_NPROCESSORS_ONLN );
    m_threads = new pthread_t[m_numThreads - 1];
    m_args = new ArgData[m_numThreads];

    // Initialize and set thread joinable
    pthread_attr_init(&m_attr);
    pthread_attr_setdetachstate(&m_attr, PTHREAD_CREATE_JOINABLE);
}


PathTracer::~PathTracer()
{
    pthread_attr_destroy(&m_attr);
    if (m_threads)
    {
        delete[] m_threads;
        m_threads = NULL;
    }
    if (m_hShapes)
    {
        delete[] m_hShapes;
        m_hShapes = NULL;
    }
    if (m_hAreaLights)
    {
        delete[] m_hAreaLights;
        m_hShapes = NULL;
    }
}


void PathTracer::init(int, const char**, GLuint, GLuint)
{
    m_initialized = true;

    // host
    m_hShapes = new Shape[MAX_SHAPES];
    m_hAreaLights = new Shape[MAX_AREA_LIGHTS];
}


void PathTracer::register2DTexture(const char *name, GLuint tex)
{
    m_textures[name] = tex;
}


void PathTracer::unregisterTexture(const char *name)
{
    m_textures.count(name);
}


void PathTracer::swapResources(const char *res1, const char *res2)
{
    GLuint temp = m_textures[res1];
    m_textures[res1] = m_textures[res2];
    m_textures[res2] = temp;
}


void PathTracer::addShape(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU)
{
    if (m_numShapes >= MAX_SHAPES)
        return;

    Shape shape;
    shape.type = type;
    shape.inv = glm::inverse(trans);
    shape.normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    shape.trans = trans;
    shape.material = material;
    shape.index = m_numShapes;

    if (sendToGPU || !sendToGPU)
        m_hShapes[m_numShapes] = shape;

    ++m_numShapes;
}


void PathTracer::addAreaLight(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU)
{
    if (m_numShapes >= MAX_SHAPES ||
            m_numAreaLights >= MAX_AREA_LIGHTS)
        return;

    Shape shape;
    shape.type = type;
    shape.inv = glm::inverse(trans);
    shape.normInv = glm::inverse(glm::transpose(glm::mat3(trans)));
    shape.trans = trans;
    shape.material = material;
    shape.index = m_numShapes;

    if (sendToGPU || !sendToGPU)
        m_hShapes[m_numShapes] = shape;
    m_hAreaLights[m_numAreaLights] = shape;

    ++m_numShapes;
    ++m_numAreaLights;
}


void PathTracer::updateShapesOnGPU()
{}


void PathTracer::setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv)
{
    m_hScaleViewInv = scaleViewInv;
    m_hEye = eye;
}


void PathTracer::tracePath(const char *writeTex, GLuint width, GLuint height, float scaleFactor)
{
    this->_tracePathCPU(writeTex, width, height, scaleFactor);
}


void *asyncPathTrace(void *args);


void PathTracer::_tracePathCPU(const char *writeTex, GLuint width, GLuint height, float scaleFactor)
{
    GLsizei widthi = static_cast<GLsizei>(width);
    GLsizei heighti = static_cast<GLsizei>(height);
    GLuint numPix = width * height;
    GLuint size = numPix * 4;

    float *data = new float[size];

    uint div = numPix / m_numThreads;
    uint end = 0;

    int rc;

    for (uint i = 0; i < m_numThreads - 1; ++i)
    {
        ArgData &argData = m_args[i];
        argData.data = data;
        argData.alpha = 1.f * scaleFactor;

        argData.start = end;
        end = (i+1) * div;
        argData.end = end;
        argData.isMainThread = false;
        rc = pthread_create(&m_threads[i], &m_attr, asyncPathTrace, reinterpret_cast<void*>(&argData));
        if (rc){
           std::cout << "ERROR: failed creating thread, " << rc << std::endl;
        }
    }
    ArgData &argData = m_args[m_numThreads-1];
    argData.data = data;
    argData.alpha = 1.f * scaleFactor;

    argData.start = 0;
    end = numPix;
    argData.end = end;
    argData.isMainThread = true;
    asyncPathTrace(reinterpret_cast<void*>(&argData));

    void *status;
    for (uint i = 0; i < m_numThreads - 1; ++i)
    {
        rc = pthread_join(m_threads[i], &status);
        if (rc){
           std::cout << "ERROR: unable to join," << rc << std::endl;
        }
    }

#ifdef USE_GRAPHICS
    glBindTexture(GL_TEXTURE_2D, m_textures[writeTex]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, widthi, heighti, 0, GL_RGBA, GL_FLOAT, data);
#else
    m_textures[writeTex];
#endif
    delete[] data;
}


void *asyncPathTrace(void *args)
{
    ArgData *argData = reinterpret_cast<ArgData*>(args);

    float *data = argData->data;
    for (GLuint i = argData->start; i < argData->end; ++i)
    {
        data[i*4  ] = 0.f;
        data[i*4+1] = 1.f;
        data[i*4+2] = 0.5f;
        data[i*4+3] = argData->alpha;
    }

    if (argData->isMainThread)
        return NULL;

    pthread_exit(NULL);
}


#endif




