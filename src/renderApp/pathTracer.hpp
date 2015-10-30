#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <unordered_map>
#include <glm/glm.hpp>

typedef unsigned int GLuint;
typedef struct cudaGraphicsResource *cudaGraphicsResource_t;

/**
 * @brief The PathTracer class.
 * Handles external CUDA functions calls
 */
class PathTracer
{

public:
    explicit PathTracer();
    virtual ~PathTracer();

    void register2DTexture(const char *name, GLuint tex);
    void unregisterTexture(const char *name);

    void setScaleViewInvEye(glm::vec4 eye, glm::mat4 scaleViewInv);
    void tracePath(const char *writeTex, GLuint width, GLuint height);

private:

    // handles OpenGL-CUDA exchanges
    std::unordered_map<const char *, cudaGraphicsResource_t> m_resources;

    float *m_dScaleViewInvEye; // device matrix
};

#endif // PATH_TRACER_H
