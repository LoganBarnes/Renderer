#include <iostream>
#include "renderApp.hpp"
#include "glHandler.hpp"
#include "pathTracer.hpp"
#include "camera.hpp"
#include "RendererConfig.hpp"


RenderApp::RenderApp()
    : m_glHandler(NULL),
      m_pathTracer(NULL),
      m_camera(NULL)
{
    m_glHandler = new GLHandler();
    m_pathTracer = new PathTracer();
    m_camera = new Camera();
}

RenderApp::~RenderApp()
{
    m_pathTracer->unregisterTexture("tex");

    if (m_glHandler)
        delete m_glHandler;
    if (m_pathTracer)
        delete m_pathTracer;
    if (m_camera)
        delete m_camera;
}

void RenderApp::init()
{
    std::string vertPath = std::string(RESOURCES_PATH) + "/shaders/default.vert";
    std::string fragPath = std::string(RESOURCES_PATH) + "/shaders/default.frag";
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

    m_glHandler->resize(640, 480, true);
    m_camera->setAspectRatio(640.f / 480.f);

    m_pathTracer->register2DTexture("tex", m_glHandler->getTexture());
    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());
}

void RenderApp::update(double)
{
//    std::cout << "updating RenderApp" << std::endl;
    m_pathTracer->tracePath("tex", m_glHandler->getViewportWidth(), m_glHandler->getViewportHeight());
}


void RenderApp::handleKeyInput()
{

}


void RenderApp::render(int interp)
{
//    std::cout << "rendering RenderApp : " << interp << std::endl;
    m_glHandler->render("default");
}


