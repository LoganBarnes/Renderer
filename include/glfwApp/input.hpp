#ifndef INPUT_HPP
#define INPUT_HPP

typedef struct GLFWwindow GLFWwindow;
class InputCallback;

/**
 * @brief The Input class
 *  singleton callback class for glfw input functions
 */
class Input
{
public:
    static Input& getInstance(); // Singleton is accessed via getInstance()

    // static methods as specified for glfw callback
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

    //this is the actual implementation of the callback method
    void defaultMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void defaultKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void defaultCursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

    void setCallback(InputCallback *callback);

private:
    Input(void) {} // private constructor necessary to allow only 1 instance

    Input(Input const&); // prevent copies
    void operator=(Input const&); // prevent assignments

    InputCallback *m_callback;
};

#endif // INPUT_HPP
