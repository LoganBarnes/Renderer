#include "GLFWApplication.hpp"
#include "renderApp.hpp"


int main(int argc, const char **argv)
{
    GLFWApplication glApp;
    glApp.setInternalApplication(new RenderApp());
    return glApp.execute(argc, argv);
}
