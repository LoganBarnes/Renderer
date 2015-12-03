#ifndef RENDER_APP_H
#define RENDER_APP_H

typedef unsigned int uint;
typedef struct GLFWwindow GLFWwindow;

class GraphicsHandler;
class PathTracer;
class Camera;

class RenderApp
{

public:
    explicit RenderApp();
    virtual ~RenderApp();

    int execute(int argc, const char **argv);

private:
    void _buildScene();
    int _runLoop();


    GraphicsHandler *m_graphics;
    PathTracer *m_pathTracer;
    Camera *m_camera;

    double m_loopFPS;
    uint m_iterationWithoutClear;
};

#endif // RENDER_APP_H
