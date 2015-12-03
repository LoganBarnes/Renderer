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

    void addShape(ShapeType type, glm::mat4 trans, glm::vec3 color);
    void addAreaLight(ShapeType type, glm::mat4 trans, glm::vec3 power, float area);

    void setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv);
    void tracePath(const char *writeTex, GLuint width, GLuint height, GLuint iteration);

private:

#ifdef USE_CUDA

    // handles OpenGL-CUDA exchanges
    std::unordered_map<const char *, cudaGraphicsResource_t> m_resources;

    void _tracePathCUDA(const char *writeTex, GLuint width, GLuint height, GLuint iteration);

    float *m_dScaleViewInvEye;  // device matrix
    Shape *m_dShapes;           // device shapes
    Shape *m_dAreaLights;   // device luminaires

    curandState *m_dRandState;

#else

    std::unordered_map<const char *, GLuint> m_textures;

    void _tracePathCPU(const char *writeTex, GLuint width, GLuint height);

    glm::mat4 m_hScaleViewInv;  // host matrix
    glm::vec4 m_hEye;
    Shape *m_hShapes;           // host shapes
    Luminaire *m_hLuminaires;   // host luminaires

#endif

    GLuint m_numShapes;
    GLuint m_numAreaLights;

    bool m_initialized;
};

#endif // PATH_TRACER_H
