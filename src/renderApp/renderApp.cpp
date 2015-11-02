#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include "renderApp.hpp"
#include "glHandler.hpp"
#include "pathTracer.hpp"
#include "camera.hpp"
#include "RendererConfig.hpp"
#include "renderObjects.hpp"


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

    _buildScene();
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


void RenderApp::_buildScene()
{
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-2.5, 0, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(0, 1, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2, 2.5, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec4(0.8, 0, 0, 1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(2.5, 0, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(0, 1, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2, 2.5, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec4(0, 0.8, 0, 1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, 2.5, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1, 0, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.001, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec4(0.8, 0.8, 0.8, 1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, -2.5, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1, 0, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.001, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec4(0.8, 0.8, 0.8, 1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, 0, -3.5));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.501, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec4(0.8, 0.8, 0.8, 1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(1, -1.5, -0.5));
    m_pathTracer->addShape(SPHERE, trans, glm::vec4(1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-1, -1.5, -2.5));
    m_pathTracer->addShape(SPHERE, trans, glm::vec4(1));
}


