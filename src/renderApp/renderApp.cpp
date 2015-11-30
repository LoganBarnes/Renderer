#include <GL/glew.h>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include "renderApp.hpp"
#include "pathTracer.hpp"
#include "renderer-config.hpp"
#include "renderObjects.hpp"

const int DEFAULT_WIDTH = 640;
const int DEFAULT_HEIGHT = 480;


#ifdef USE_GRAPHICS
#include "graphicsHandler.hpp"
#include "camera.hpp"
#endif


RenderApp::RenderApp()
    : m_graphics(NULL),
      m_pathTracer(NULL),
      m_camera(NULL)
{
#ifdef USE_GRAPHICS
    m_graphics = new GraphicsHandler(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    m_camera = new Camera();
#endif
    m_pathTracer = new PathTracer();
}

RenderApp::~RenderApp()
{
#ifdef USE_GRAPHICS
    m_pathTracer->unregisterTexture("tex");

    if (m_camera)
        delete m_camera;
    if (m_graphics)
        delete m_graphics;
#endif
    if (m_pathTracer)
        delete m_pathTracer;
}

//int RenderApp::execute(int , const char **)
int RenderApp::execute(int argc, const char **argv)
{
    m_pathTracer->init(argc, argv, DEFAULT_WIDTH, DEFAULT_HEIGHT);

#ifdef USE_GRAPHICS
    if (!m_graphics->init("Render App"))
        return 1;

    std::string vertPath = std::string(RESOURCES_PATH) + "/shaders/default.vert";
    std::string fragPath = std::string(RESOURCES_PATH) + "/shaders/default.frag";
    m_graphics->addProgram("default", vertPath.c_str(), fragPath.c_str());

    GLfloat screenData[8] = {-1, 1, -1, -1, 1, 1, 1, -1};
    m_graphics->addUVBuffer("fullscreen", "default", screenData, 8);

    m_graphics->addTextureArray("tex", DEFAULT_WIDTH, DEFAULT_HEIGHT, NULL);

    m_camera->setAspectRatio(
                static_cast<float>(DEFAULT_WIDTH) /
                static_cast<float>(DEFAULT_HEIGHT));

    m_pathTracer->register2DTexture("tex", m_graphics->getTexture("tex"));
    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());
#endif

    _buildScene();

    return this->_runLoop();
}


int RenderApp::_runLoop()
{
#ifdef USE_GRAPHICS
    while (!m_graphics->checkWindowShouldClose())
    {
        m_pathTracer->tracePath("tex", DEFAULT_WIDTH, DEFAULT_HEIGHT);
        m_graphics->render("default", "fullscreen", 4, GL_TRIANGLE_STRIP, "tex");
        m_graphics->updateWindow();
    }
#endif

    return 0;
}


//void RenderApp::update(double)
//{
//    m_pathTracer->tracePath("tex", DEFAULT_WIDTH, DEFAULT_HEIGHT);
//}


//void RenderApp::handleKeyInput()
//{

//}


//void RenderApp::render(int)
//{
//    m_graphics->render("default", "fullscreen", 4, GL_TRIANGLE_STRIP, "tex");
//}


void RenderApp::_buildScene()
{
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-2.5, 0, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(0, 1, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2, 2.5, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec3(0.8, 0, 0));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(2.5, 0, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(0, 1, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2, 2.5, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec3(0, 0.8, 0));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, 2.5, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1, 0, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.001, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec3(0.8, 0.8, 0.8));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, -2.5, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1, 0, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.001, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec3(0.8, 0.8, 0.8));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, 0, -3.5));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501, 2.501, 1));
    m_pathTracer->addShape(QUAD, trans, glm::vec3(0.8, 0.8, 0.8));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(1, -1.5, -0.5));
    m_pathTracer->addShape(SPHERE, trans, glm::vec3(1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-1, -1.5, -2.5));
    m_pathTracer->addShape(SPHERE, trans, glm::vec3(1));

    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0, 2.49, -1.5));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1, 0, 0));
    trans *= glm::scale(glm::mat4(), glm::vec3(.5, .3, 1));
    m_pathTracer->addAreaLight(QUAD, trans, glm::vec3(4));
}


