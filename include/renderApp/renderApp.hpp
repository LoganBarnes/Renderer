#ifndef RENDER_APP_H
#define RENDER_APP_H

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
};

#endif // RENDER_APP_H
