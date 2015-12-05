#ifndef INPUT_HPP
#define INPUT_HPP

// Input.h (the actual callback class for glfwSetMouseButtonCallback)
class Input
{
public:
    static Input& getInstance(); // Singleton is accessed via getInstance()

    static void mouseButtonCallback(int key, int action); // this method is specified as glfw callback

    void mouseButtonCallbackImpl(int key, int action); //this is the actual implementation of the callback method

private:
    Input(void) {} // private constructor necessary to allow only 1 instance

    Input(Input const&); // prevent copies
    void operator=(Input const&); // prevent assignments
};

#endif // INPUT_HPP
