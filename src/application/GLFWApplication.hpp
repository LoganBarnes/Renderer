#ifndef GLFW_APPLICATION_H
#define GLFW_APPLICATION_H

#include <vector>

typedef struct GLFWwindow GLFWwindow;
typedef void (* GLFWerrorfun)(int,const char*);
typedef void (* GLFWkeyfun)(GLFWwindow*,int,int,int,int);

class Application;

class GLFWApplication
{

public:
    explicit GLFWApplication();
    ~GLFWApplication();

    void setInternalApplication(Application *app);
    void setWindowSize(int width, int height);

    int execute(const char *title = "GLFW Application");

    static void key_callback(GLFWwindow* window, int key, int, int action, int);
    static void error_callback(int error, const char* description);

private:
    bool _initGLFW(const char *title);
    void _terminateGLFW();

    Application *m_app;

    GLFWwindow *m_window;
    int m_width, m_height;

    double m_fps;

    bool m_initialized;
};

#endif // GLFW_APPLICATION_H
