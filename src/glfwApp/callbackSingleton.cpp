#include "callbackSingleton.hpp"
#include "callback.hpp"


CallbackSingleton& CallbackSingleton::getInstance() // Singleton is accessed via getInstance()
{
    static CallbackSingleton instance; // lazy singleton, instantiated on first use
    return instance;
}


/*
 *
 * STATIC CALLBACKS
 *
 */

/**
 * @brief error_callback
 * @param error
 * @param description
 */
void CallbackSingleton::errorCallback(int error, const char* description)
{
    CallbackSingleton::getInstance().defaultErrorCallback(error, description);
}

/**
 * @brief window_size_callback
 * @param window
 * @param width
 * @param height
 */
void CallbackSingleton::windowSizeCallback(GLFWwindow* window, int width, int height)
{
    CallbackSingleton::getInstance().defaultWindowSizeCallback(window, width, height);
}

/**
 * @brief Input::keyCallback
 * @param window
 * @param key
 * @param scancode
 * @param action
 * @param mods
 */
void CallbackSingleton::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    CallbackSingleton::getInstance().defaultKeyCallback(window, key, scancode, action, mods);
}

/**
 * @brief Input::cursorPositionCallback
 * @param window
 * @param xpos
 * @param ypos
 */
void CallbackSingleton::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    CallbackSingleton::getInstance().defaultCursorPositionCallback(window, xpos, ypos);
}

/**
 * @brief Input::mouseButtonCallback
 * @param window
 * @param button
 * @param action
 * @param mods
 */
void CallbackSingleton::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    CallbackSingleton::getInstance().defaultMouseButtonCallback(window, button, action, mods);
}


/*
 *
 * ACTUAL MEMBER IMPLEMENTATIONS
 *
 */

/**
 * @brief CallbackSingleton::defaultErrorCallback
 * @param error
 * @param description
 */
void CallbackSingleton::defaultErrorCallback(int error, const char *description)
{
    if (m_callback)
        m_callback->handleError(error, description);
}

/**
 * @brief CallbackSingleton::defaultWindowSizeCallback
 * @param window
 * @param width
 * @param height
 */
void CallbackSingleton::defaultWindowSizeCallback(GLFWwindow *window, int width, int height)
{
    if (m_callback)
        m_callback->handleWindowSize(window, width, height);
}

/**
 * @brief Input::defaultKeyCallback
 * @param window
 * @param key
 * @param scancode
 * @param action
 * @param mods
 */
void CallbackSingleton::defaultKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (m_callback)
        m_callback->handleKey(window, key, scancode, action, mods);
}

/**
 * @brief Input::defaultCursorPosition_callback
 * @param window
 * @param xpos
 * @param ypos
 */
void CallbackSingleton::defaultCursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (m_callback)
        m_callback->handleCursorPosition(window, xpos, ypos);
}

/**
 * @brief Input::defaultMouseButtonCallback
 * @param window
 * @param button
 * @param action
 * @param mods
 */
void CallbackSingleton::defaultMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (m_callback)
        m_callback->handleMouseButton(window, button, action, mods);
}



void CallbackSingleton::setCallback(Callback *callback)
{
    m_callback = callback;
}






