#include <GLFW/glfw3.h>
#include <iostream>
#include "renderCallback.hpp"
#include "renderApp.hpp"


RenderInput::RenderInput(RenderApp *app, GLFWwindow *window)
    : m_app(app),
      m_leftMouseDown(false),
      m_rightMouseDown(false),
      m_shiftDown(false)
{
    glfwGetCursorPos(window, &m_prevX, &m_prevY);
}


RenderInput::~RenderInput()
{}


void RenderInput::handleError(int error, const char *description)
{
    std::cerr << "ERROR: (" << error << ") " << description << std::endl;
}


////RenderInput::handleWindowSize(GLFWwindow *window, int width, int height)
//void RenderInput::handleWindowSize(GLFWwindow*, int width, int height)
//{
//    m_app->resize(width, height);
//}


//void RenderInput::handleMouseButton(GLFWwindow *window, int button, int action, int mods)
void RenderInput::handleMouseButton(GLFWwindow*, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_1)
    {
        if (action == GLFW_PRESS)
            m_leftMouseDown = true;
        else
            m_leftMouseDown = false;
    }
    else if (button == GLFW_MOUSE_BUTTON_2)
    {
        if (action == GLFW_PRESS)
            m_rightMouseDown = true;
        else
            m_rightMouseDown = false;
    }
}


void RenderInput::handleKey(GLFWwindow *window, int key, int, int action, int)
{
    switch(key)
    {
    case GLFW_KEY_ESCAPE:

        if (action == GLFW_RELEASE)
            glfwSetWindowShouldClose(window, GL_TRUE);
        break;

    case GLFW_KEY_LEFT_SHIFT:
    case GLFW_KEY_RIGHT_SHIFT:

        if (action == GLFW_PRESS)
            m_shiftDown = true;
        else
            m_shiftDown = false;
        break;

    default:
        break;
    }

//     std::cout << key << std::endl;
}


//void RenderInput::handleCursorPosition(GLFWwindow *window, double xpos, double ypos)
void RenderInput::handleCursorPosition(GLFWwindow*, double xpos, double ypos)
{
    if (m_leftMouseDown)
    {
        m_app->rotateCamera(m_prevX - xpos, m_prevY - ypos);
    }

    m_prevX = xpos;
    m_prevY = ypos;
//    std::cout << xpos << ", " << ypos << std::endl;
}



