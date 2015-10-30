#ifndef RENDER_APP_H
#define RENDER_APP_H

#include "Application.hpp"

class GLHandler;
class PathTracer;
class Camera;

class RenderApp : public Application
{

public:
    explicit RenderApp();
    virtual ~RenderApp();

    virtual void init();

    virtual void update(double deltaTime);
    virtual void handleKeyInput();

    virtual void render(int interp = 1.0);

private:
    GLHandler *m_glHandler;
    PathTracer *m_pathTracer;
    Camera *m_camera;
};

#endif // RENDER_APP_H
