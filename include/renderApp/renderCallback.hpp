#ifndef RENDER_INPUT_HPP
#define RENDER_INPUT_HPP

#include "callback.hpp"

class RenderApp;

class RenderInput : public Callback
{
public:
    RenderInput(RenderApp *app, GLFWwindow *window);
    virtual ~RenderInput();

    virtual void handleError(int error, const char* description);
//    virtual void handleWindowSize(GLFWwindow* window, int width, int height);

    virtual void handleMouseButton(GLFWwindow* window, int button, int action, int mods);
    virtual void handleKey(GLFWwindow* window, int key, int scancode, int action, int mods);
    virtual void handleCursorPosition(GLFWwindow* window, double xpos, double ypos);

private:
    RenderApp *m_app;

    bool m_leftMouseDown;
    bool m_rightMouseDown;

    bool m_shiftDown;

    double m_prevX;
    double m_prevY;

};

#endif // RENDER_INPUT_HPP
