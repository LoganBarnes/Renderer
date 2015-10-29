#include <iostream>
#include "renderApp.hpp"
#include "glHandler.hpp"
#include "RendererConfig.hpp"




RenderApp::RenderApp()
    : m_glHandler(NULL)
{
    m_glHandler = new GLHandler();
}

RenderApp::~RenderApp()
{
    if (m_glHandler)
        delete m_glHandler;
}

void RenderApp::init()
{
    std::string vertPath = std::string(RESOURCES_PATH) + "shaders/default.vert";
    std::string fragPath = std::string(RESOURCES_PATH) + "shaders/default.frag";
    m_glHandler->addProgram("default", vertPath.c_str(), fragPath.c_str());

    GLfloat *data = new GLfloat[8]();
    data[0] = -1;
    data[1] =  1;
    data[2] = -1;
    data[3] = -1;
    data[4] =  1;
    data[5] =  1;
    data[6] =  1;
    data[7] = -1;

    m_glHandler->setBuffer("default", data);
    delete[] data;
}

void RenderApp::update(double)
{
//    std::cout << "updating RenderApp" << std::endl;
}


void RenderApp::handleKeyInput()
{

}


void RenderApp::render(int interp)
{
//    std::cout << "rendering RenderApp : " << interp << std::endl;
    m_glHandler->render("default");
}


