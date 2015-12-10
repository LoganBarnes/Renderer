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

    void rotateCamera(double deltaX, double deltaY);
    void zoomCamera(double deltaZ);
    void resize(int width, int height);

    int execute(int argc, const char **argv);

private:
    void _buildScene();
    int _runLoop();

    void _resetBlendTexture();

    void _render(const char *program, const char *mainTex, int iteration, bool texSize = true, const char *blendTex = 0);

    GraphicsHandler *m_graphics;
    PathTracer *m_pathTracer;
    Camera *m_camera;

    int m_texWidth;
    int m_texHeight;

    double m_loopFPS;
    int m_iterationWithoutClear;
};

#endif // RENDER_APP_H
