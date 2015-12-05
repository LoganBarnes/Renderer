#include "input.hpp"


Input& Input::getInstance() // Singleton is accessed via getInstance()
{
    static Input instance; // lazy singleton, instantiated on first use
    return instance;
}


void Input::mouseButtonCallback(int key, int action) // this method is specified as glfw callback
{
    //here we access the instance via the singleton pattern and forward the callback to the instance method
    Input::getInstance().mouseButtonCallbackImpl(key, action);
}

//void Input::mouseButtonCallbackImpl(int key, int action) //this is the actual implementation of the callback method
void Input::mouseButtonCallbackImpl(int , int ) //this is the actual implementation of the callback method
{
    //the callback is handled in this instance method
    //... [CODE here]
}
