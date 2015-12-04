#ifndef RENDER_APP_H
#define RENDER_APP_H

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

    void _render(const char *program, const char *mainTex, int iteration, const char *blendTex = 0);

    GraphicsHandler *m_graphics;
    PathTracer *m_pathTracer;
    Camera *m_camera;

    double m_loopFPS;
    int m_iterationWithoutClear;
};

#endif // RENDER_APP_H
