#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <unordered_map>
#include "glm/glm.hpp"
#include "RenderTypes.hpp"
#include "RendererConfig.hpp"

typedef unsigned int GLuint;


#ifdef USE_CUDA
#include <curand_kernel.h>
typedef struct cudaGraphicsResource*cudaGraphicsResource_t;
#else
#include <pthread.h>
struct ThreadData;

#endif

struct Shape;

struct Material;


namespace rndr
{


/**
 * @brief The PathTracer class.
 * Handles external CUDA functions calls
 */
class PathTracer
{

public:

  explicit
  PathTracer( );

  ~PathTracer( );

  void init (
             GLuint width,
             GLuint height
             );

  void register2DTexture (
                          const std::string name,
                          GLuint            tex
                          );
  void unregisterTexture ( const std::string name );
  void swapResources (
                      const std::string res1,
                      const std::string res2
                      );

  void addShape (
                 ShapeType type,
                 glm::mat4 trans,
                 Material  material,
                 bool      sendToGPU = true
                 );
  void addAreaLight (
                     ShapeType type,
                     glm::mat4 trans,
                     Material  material,
                     bool      sendToGPU = true
                     );
  void updateShapesOnGPU ( );

  void setScaleViewInvEye (
                           glm::vec4 eye,
                           glm::mat4 scaleViewInv
                           );
  void tracePath (
                  const std::string writeTex,
                  GLuint            width,
                  GLuint            height,
                  float             scaleFactor
                  );


private:

  Shape *m_hShapes;         // host shapes
  Shape *m_hAreaLights;     // host luminaires

#ifdef USE_CUDA

  // handles OpenGL-CUDA exchanges
  std::unordered_map< std::string, cudaGraphicsResource_t > m_resources;

  void _tracePathCUDA (
                       const std::string writeTex,
                       GLuint            width,
                       GLuint            height,
                       float             scaleFactor
                       );

  float *m_dScaleViewInvEye;    // device matrix
  Shape *m_dShapes;             // device shapes
  Shape *m_dAreaLights;     // device luminaires

  curandState *m_dRandState;

#else // ifdef USE_CUDA

  std::unordered_map< const char*, GLuint > m_textures;

  pthread_t *m_threads;
  ThreadData *m_args;

  pthread_attr_t m_attr;
  GLuint m_numThreads;

  glm::mat4 m_hScaleViewInv;    // host matrix
  glm::vec4 m_hEye;

  void _tracePathCPU (
                      const std::string writeTex,
                      GLuint            width,
                      GLuint            height,
                      float             scaleFactor
                      );

#endif // ifdef USE_CUDA

  GLuint m_numShapes;
  GLuint m_numAreaLights;

  bool m_initialized;
};


} // namespace rndr


#endif // PATH_TRACER_H
