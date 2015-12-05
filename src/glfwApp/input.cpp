#include "input.hpp"
#include "inputCallback.hpp"


Input& Input::getInstance() // Singleton is accessed via getInstance()
{
    static Input instance; // lazy singleton, instantiated on first use
    return instance;
}


/*
 *
 * STATIC CALLBACKS
 *
 */

/**
 * @brief Input::keyCallback
 * @param window
 * @param key
 * @param scancode
 * @param action
 * @param mods
 */
void Input::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Input::getInstance().defaultKeyCallback(window, key, scancode, action, mods);
}

/**
 * @brief Input::cursorPositionCallback
 * @param window
 * @param xpos
 * @param ypos
 */
void Input::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    Input::getInstance().defaultCursorPositionCallback(window, xpos, ypos);
}

/**
 * @brief Input::mouseButtonCallback
 * @param window
 * @param button
 * @param action
 * @param mods
 */
void Input::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    Input::getInstance().defaultMouseButtonCallback(window, button, action, mods);
}


/*
 *
 * ACTUAL MEMBER IMPLEMENTATIONS
 *
 */

/**
 * @brief Input::defaultKeyCallback
 * @param window
 * @param key
 * @param scancode
 * @param action
 * @param mods
 */
void Input::defaultKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (m_callback)
        m_callback->onKey(window, key, scancode, action, mods);
}

/**
 * @brief Input::defaultCursorPosition_callback
 * @param window
 * @param xpos
 * @param ypos
 */
void Input::defaultCursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (m_callback)
        m_callback->onCursorPosition(window, xpos, ypos);
}

/**
 * @brief Input::defaultMouseButtonCallback
 * @param window
 * @param button
 * @param action
 * @param mods
 */
void Input::defaultMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (m_callback)
        m_callback->onMouseButton(window, button, action, mods);
}



void Input::setCallback(InputCallback *callback)
{
    m_callback = callback;
}






