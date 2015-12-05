#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <unordered_map>
#include <glm/glm.hpp>
#include "renderTypes.hpp"
#include "renderer-config.hpp"

typedef unsigned int GLuint;

#ifdef USE_CUDA
#include <curand_kernel.h>
typedef struct cudaGraphicsResource *cudaGraphicsResource_t;
#endif

struct Shape;
struct Material;

/**
 * @brief The PathTracer class.
 * Handles external CUDA functions calls
 */
class PathTracer
{

public:
    explicit PathTracer();
    virtual ~PathTracer();

    void init(int argc, const char **argv, GLuint width, GLuint height);

    void register2DTexture(const char *name, GLuint tex);
    void unregisterTexture(const char *name);
    void swapResources(const char *res1, const char *res2);

    void addShape(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU = true);
    void addAreaLight(ShapeType type, glm::mat4 trans, Material material, bool sendToGPU = true);
    void updateShapesOnGPU();

    void setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv);
    void tracePath(const char *writeTex, GLuint width, GLuint height, float scaleFactor);

private:

    Shape *m_hShapes;           // host shapes
    Shape *m_hAreaLights;   // host luminaires

#ifdef USE_CUDA

    // handles OpenGL-CUDA exchanges
    std::unordered_map<const char *, cudaGraphicsResource_t> m_resources;

    void _tracePathCUDA(const char *writeTex, GLuint width, GLuint height, float scaleFactor);

    float *m_dScaleViewInvEye;  // device matrix
    Shape *m_dShapes;           // device shapes
    Shape *m_dAreaLights;   // device luminaires

    curandState *m_dRandState;

#else

    std::unordered_map<const char *, GLuint> m_textures;

    glm::mat4 m_hScaleViewInv;  // host matrix
    glm::vec4 m_hEye;

    void _tracePathCPU(const char *writeTex, GLuint width, GLuint height);

#endif

    GLuint m_numShapes;
    GLuint m_numAreaLights;

    bool m_initialized;
};

#endif // PATH_TRACER_H
