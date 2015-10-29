#ifndef RENDER_APP_H
#define RENDER_APP_H

#include "Application.hpp"

//typedef void (* GLFWerrorfun)(int,const char*);
//typedef struct GLFWwindow GLFWwindow;

class GLHandler;

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

};

#endif // RENDER_APP_H
