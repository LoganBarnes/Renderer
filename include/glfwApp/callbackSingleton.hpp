#ifndef INPUT_HPP
#define INPUT_HPP

typedef struct GLFWwindow GLFWwindow;
class Callback;

/**
 * @brief The Input class
 *  singleton callback class for glfw input functions
 */
class CallbackSingleton
{
public:
    static CallbackSingleton& getInstance(); // Singleton is accessed via getInstance()

    // basic static callback functions
    static void errorCallback(int error, const char* description);
    static void windowSizeCallback(GLFWwindow* window, int width, int height);

    // input static callback functions
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    // the actual implementations of the callback methods
    void defaultErrorCallback(int error, const char* description);
    void defaultWindowSizeCallback(GLFWwindow* window, int width, int height);

    void defaultMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void defaultKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void defaultCursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    void defaultScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    void setCallback(Callback *callback);

private:
    CallbackSingleton(void) {} // private constructor necessary to allow only 1 instance

    CallbackSingleton(CallbackSingleton const&); // prevent copies
    void operator=(CallbackSingleton const&); // prevent assignments

    Callback *m_callback;

    /*
     *  C++ 11
     * =======
     * Delete unwanted methods instead
     *
     * CallbackSingleton(CallbackSingleton const&)     = delete;
     * void operator=(CallbackSingleton const&)        = delete;
     */
};

#endif // INPUT_HPP
