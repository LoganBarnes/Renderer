#ifndef RENDER_INPUT_HPP
#define RENDER_INPUT_HPP

#include "inputCallback.hpp"

class RenderInput : public InputCallback
{
public:
    RenderInput();
    virtual ~RenderInput();

    //this is the actual implementation of the callback method
    virtual void onMouseButton(GLFWwindow* window, int button, int action, int mods);
    virtual void onKey(GLFWwindow* window, int key, int scancode, int action, int mods);
    virtual void onCursorPosition(GLFWwindow* window, double xpos, double ypos);

};

#endif // RENDER_INPUT_HPP
