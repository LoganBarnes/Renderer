#include "GLFWApplication.hpp"
#include "renderApp.hpp"


int main(void)
{
    GLFWApplication glApp;
    glApp.setInternalApplication(new RenderApp());
    return glApp.execute();
}
