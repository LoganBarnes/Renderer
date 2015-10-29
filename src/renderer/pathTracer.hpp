#ifndef PATH_TRACER_H
#define PATH_TRACER_H

typedef unsigned int GLuint;
typedef float GLfloat;

/**
 * @brief The PathTracer class.
 * Handles external CUDA functions calls
 */
class PathTracer
{

public:
    explicit PathTracer();
    virtual ~PathTracer();

};

#endif // PATH_TRACER_H
