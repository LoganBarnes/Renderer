#ifndef INPUT_CALLBACK_HPP
#define INPUT_CALLBACK_HPP

typedef struct GLFWwindow GLFWwindow;


class InputCallback
{
public:
    virtual ~InputCallback() {}

    //this is the actual implementation of the callback method
    virtual void onMouseButton(GLFWwindow* window, int button, int action, int mods) = 0;
    virtual void onKey(GLFWwindow* window, int key, int scancode, int action, int mods) = 0;
    virtual void onCursorPosition(GLFWwindow* window, double xpos, double ypos) = 0;

};

#endif // INPUT_CALLBACK_HPP
