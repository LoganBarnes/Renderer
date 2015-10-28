#include <GLFW/glfw3.h>
#include <assert.h>
#include <iostream>
#include "renderer.hpp"
#include "RendererConfig.hpp" // generated by cmake

#ifdef USE_CUDA
#include "CudaFunctions.cuh"
#endif

Renderer::Renderer(int argc, const char** argv, ulong max)
    : m_max(max)
{
    // create a host array and initialize it to {1, 2, 3, ..., m_max}
    ulong hNumbers[m_max];
    for (ulong i = 0; i < m_max; i++)
    {
        hNumbers[i] = i + 1;
    }

    // CUDA FUNCTIONS:
#ifdef USE_CUDA
    cudaInit(argc, argv); // initialiaze the cuda device
    allocateArray((void**)&m_dNumbers, m_max*sizeof(ulong)); // allocate device array
    copyArrayToDevice(m_dNumbers, hNumbers, 0, m_max*sizeof(ulong)); // copy host array to device array
#endif

}

Renderer::~Renderer()
{
    // CUDA FUNCTION: free device memory
#ifdef USE_CUDA
    freeArray(m_dNumbers);
    cudaDestroy();
#endif
}


bool Renderer::initGLFW(GLFWerrorfun error_callback)
{
    if (!glfwInit())
        return false;

    glfwSetErrorCallback(error_callback);
    
    return true;
}


void Renderer::terminateGLFW()
{
    for(std::vector<GLFWwindow*>::iterator it = m_windows.begin(); it != m_windows.end(); ++it) {
         glfwDestroyWindow(*it);
    }
    glfwTerminate();
}


bool Renderer::createWindow(int width, int height, const char* title)
{
    GLFWwindow *window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window)
        return false;

    m_windows.push_back(window);

    return true;
}


// compute the sum of 1 + 2 + 3 + ... + number
ulong Renderer::sumNumber(ulong number)
{
    
    // CUDA FUNCTION:
#ifdef USE_CUDA
    ulong sum = sumNumbers(m_dNumbers, number);
#else
    ulong sum = 0;

    for (uint i = 1; i <= number; ++i)
    {
        sum += i;
    }

    // check that the sum is correct
    assert(sum == ( (number * (number + 1) ) / 2 ) );
#endif

    return sum;
}


void Renderer::setMaxNumber(ulong max)
{
    m_max = max;
}



