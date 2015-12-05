#include <GLFW/glfw3.h>
#include "renderInput.hpp"


RenderInput::RenderInput()
{}


RenderInput::~RenderInput()
{}


//void RenderInput::onMouseButton(GLFWwindow *window, int button, int action, int mods)
void RenderInput::onMouseButton(GLFWwindow*, int, int, int)
{
}


void RenderInput::onKey(GLFWwindow *window, int key, int, int action, int)
{
     if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
         glfwSetWindowShouldClose(window, GL_TRUE);
}


//void RenderInput::onCursorPosition(GLFWwindow *window, double xpos, double ypos)
void RenderInput::onCursorPosition(GLFWwindow*, double, double)
{
}



