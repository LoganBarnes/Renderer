#ifndef RENDERER_H
#define RENDERER_H

#include <vector>

typedef unsigned int uint;
typedef unsigned long ulong;
typedef void (* GLFWerrorfun)(int,const char*);
typedef struct GLFWwindow GLFWwindow;

class Renderer
{

public:
    explicit Renderer(int argc, const char** argv, ulong max);
    ~Renderer();

    bool initGLFW(GLFWerrorfun error_callback);
    void terminateGLFW();

    bool createWindow(int width, int height, const char *title);

    ulong sumNumber(ulong number);
    void setMaxNumber(ulong max);

private:
    ulong *m_dNumbers; // device array
    ulong m_max;

    std::vector<GLFWwindow*> m_windows;
};

#endif // RENDERER_H
