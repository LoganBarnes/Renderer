#ifndef CPU_RENDER_HPP
#define CPU_RENDER_HPP

#include <glm.hpp>
#include <pthread.h>
#include <unistd.h>

typedef unsigned int uint;
struct Shape;
struct ThreadData;

void cpu_init(uint *numThreads, pthread_t **threads, pthread_attr_t *attr)
{
    *numThreads = sysconf( _SC_NPROCESSORS_ONLN );
    *threads = new pthread_t[*numThreads - 1];

    // Initialize and set thread joinable
    pthread_attr_init(attr);
    pthread_attr_setdetachstate(attr, PTHREAD_CREATE_JOINABLE);
}


void cpu_destroy(pthread_t **threads, pthread_attr_t *attr)
{
    pthread_attr_destroy(attr);
    if (*threads)
    {
        delete[] *threads;
        *threads = NULL;
    }
}


/*
 * from 'pathTracer.cpp'
 */
void cpu_tracePath(float *data,
                   glm::mat4 scaleViewInv,
                   glm::vec4 eye,
                   Shape *shapes,
                   uint numShapes,
                   Shape *areaLights,
                   uint numAreaLights,
                   glm::ivec3 texDim,
                   uint numThreads,
                   pthread_t *threads,
                   pthread_attr_t attr,
                   ThreadData *args,
                   bool debugEmit,
                   bool debugDirect,
                   bool debugIndirect,
                   int bounceLimit = 1000,
                   float scale = 1.f);

#endif // CPU_RENDER_HPP
