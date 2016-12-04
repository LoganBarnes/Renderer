#include <GL/glew.h>
#include <iostream>
#include "gtc/matrix_transform.hpp"
#include "renderApp.hpp"
#include "pathTracer.hpp"
#include "renderer-config.hpp"
#include "renderObjects.hpp"
#include "graphicsHandler.hpp"
#include "camera.hpp"
#include "renderCallback.hpp"

const int DEFAULT_WIDTH  = 640;
const int DEFAULT_HEIGHT = 480;

//const int DEFAULT_WIDTH = 480;
//const int DEFAULT_HEIGHT = 380;

const float LIGHT_SCALING = 0.5f;


#ifdef USE_CUDA
float3 modFloat3(float x) { return make_float3(x); }
float3 modFloat3(float x, float y, float z) { return make_float3(x, y, z); }
#else
glm::vec3 modFloat3(float x) { return glm::vec3(x); }
glm::vec3 modFloat3(float x, float y, float z) { return glm::vec3(x, y, z); }
#endif


RenderApp::RenderApp()
    : m_graphics(NULL),
      m_pathTracer(NULL),
      m_camera(NULL),
      m_loopFPS(60.0),
      m_iterationWithoutClear(2)
{
    m_graphics = new GraphicsHandler(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    m_camera = new Camera();
    m_pathTracer = new PathTracer();
}

RenderApp::~RenderApp()
{
    m_pathTracer->unregisterTexture("currTex");

    if (m_camera)
        delete m_camera;
    if (m_graphics)
        delete m_graphics;
    if (m_pathTracer)
        delete m_pathTracer;
}


void RenderApp::rotateCamera(double deltaX, double deltaY)
{
    m_camera->updateOrbit(0.f, static_cast<float>(deltaX), static_cast<float>(deltaY));
    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());

    this->_resetBlendTexture();
}


void RenderApp::zoomCamera(double deltaZ)
{
    m_camera->updateOrbit(static_cast<float>(deltaZ * 0.25), 0.f, 0.f);
    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());

    this->_resetBlendTexture();
}


//void RenderApp::resize(int width, int height)
//{
//    m_graphics->resize(width, height);
//    m_camera->setAspectRatio(
//                static_cast<float>(width) / static_cast<float>(height));
//    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());

//    this->_resetBlendTexture();

//    // render texture
//    m_graphics->bindFramebuffer(NULL);
//    this->_render("default", "blendTex2", -1, false);

//    // swap buffers
//    m_graphics->updateWindow();
//}


int RenderApp::execute(int argc, const char **argv)
{
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;

    bool antiAliasing = false;

    if (argc > 1)
    {
        if (!std::strcmp("-a", argv[1]))
            antiAliasing = true;
        else
        {
            width = strtoul(argv[1], NULL, 0);
            height = width;
        }
    }
    if (argc > 2)
    {
        if (!std::strcmp("-a", argv[2]))
            antiAliasing = true;
        else
            height = strtoul(argv[2], NULL, 0);
    }
    if (argc > 3 && !std::strcmp("-a", argv[3]))
        antiAliasing = true;

    m_texWidth = width;
    m_texHeight = height;

    if (antiAliasing)
    {
        m_texWidth *= 2;
        m_texHeight *= 2;
    }

    uint texWu = static_cast<uint>(m_texWidth);
    uint texHu = static_cast<uint>(m_texHeight);

    if (!m_graphics->init("Render App"))
        return 1;

    RenderInput input(this, m_graphics->getWindow());
    m_graphics->setCallback(&input);

    m_graphics->setWindowSize(width, height);

    std::string vertPath = std::string(RESOURCES_PATH) + "/shaders/default.vert";
    std::string fragPath = std::string(RESOURCES_PATH) + "/shaders/default.frag";
    m_graphics->addProgram("default", vertPath.c_str(), fragPath.c_str());

    GLfloat screenData[8] = {-1, 1, -1, -1, 1, 1, 1, -1};
    m_graphics->addUVBuffer("fullscreen", "default", screenData, 8);

    m_graphics->addTextureArray("currTex", m_texWidth, m_texHeight);
    m_graphics->addTextureArray("blendTex1", m_texWidth, m_texHeight, NULL, true);
    m_graphics->addTextureArray("blendTex2", m_texWidth, m_texHeight, NULL, true);
    m_graphics->addFramebuffer("framebuffer1", m_texWidth, m_texWidth, "blendTex1");
    m_graphics->addFramebuffer("framebuffer2", m_texWidth, m_texWidth, "blendTex2");

    m_camera->setAspectRatio(
                static_cast<float>(width) / static_cast<float>(height));
    m_camera->updateOrbit(7.f, 0.f, 0.f);

    m_pathTracer->init(argc, argv, texWu, texHu);

    m_pathTracer->register2DTexture("currTex", m_graphics->getTexture("currTex"));
    m_pathTracer->setScaleViewInvEye(m_camera->getEye(), m_camera->getScaleViewInvMatrix());

    _buildScene();
    m_pathTracer->updateShapesOnGPU();

    std::cout << "Rendering..." << std::endl;
    return this->_runLoop();
}


int RenderApp::_runLoop()
{
//    uint counterMax = 200000000;
//    uint counterMax = 1000000;
//    uint counterMax = 1000;
//    uint counterMax = 10;
//    uint counter = counterMax + 1;

    // TODO: stop rendering after convergence limit?

    this->_resetBlendTexture();

    while (!m_graphics->checkWindowShouldClose())
    {
        // temporary timer
//        if (counter < counterMax)
//        {
//            ++counter;
//            continue;
//        }
//        else
//            counter = 0;

        // blend to texture
        m_graphics->bindFramebuffer("framebuffer2");
        m_pathTracer->tracePath("currTex", static_cast<uint>(m_texWidth), static_cast<uint>(m_texHeight), LIGHT_SCALING);
        this->_render("default", "currTex", m_iterationWithoutClear++, true, "blendTex1");

        // render texture
        m_graphics->bindFramebuffer(NULL);
        this->_render("default", "blendTex2", -1, false);

        // swap texture and framebuffer values so we can use the same names
        m_graphics->swapTextures("blendTex1", "blendTex2");
        m_graphics->swapFramebuffers("framebuffer1", "framebuffer2");

        // swap buffers and trigger callback functions
        m_graphics->updateWindow();
    }

    return 0;
}


void RenderApp::_resetBlendTexture()
{
    // clear blending texture
    m_graphics->bindFramebuffer("framebuffer1");
    m_graphics->clearWindow(m_texWidth, m_texHeight);

    // reset iteration number
    m_iterationWithoutClear = 1;
}


void RenderApp::_render(const char *program, const char *mainTex, int iteration, bool texSize, const char *blendTex)
{
    if (texSize)
        m_graphics->clearWindow(m_texWidth, m_texHeight);
    else
        m_graphics->clearWindow();
    m_graphics->useProgram(program);
    m_graphics->setTextureUniform(program, "uTexture", mainTex, 0);
    if (blendTex)
        m_graphics->setTextureUniform(program, "uBlendTex", blendTex, 1);
    m_graphics->setIntUniform(program, "uIteration", iteration);
    m_graphics->renderBuffer("fullscreen", 4, GL_TRIANGLE_STRIP);
}


void RenderApp::_buildScene()
{
    // 0: left (green) wall
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-2.5f, 0.f, 0.f));
    trans *= glm::rotate(glm::mat4(), glm::radians(-90.f), glm::vec3(0.f, 1.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.5f, 2.5f, 1.f));

    Material mat;
    mat.power = modFloat3(0.f);
    mat.emitted = modFloat3(0.f);
    mat.color = modFloat3(0.117f, 0.472f, 0.115f);
    mat.lambertianReflect = modFloat3(0.117f, 0.472f, 0.115f);
    mat.etaPos = 1.f;
    mat.etaNeg = 1.f;
    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 1: right (red) wall
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(2.5f, 0.f, 0.f));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(0.f, 1.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.5f, 2.5f, 1.f));

    mat.color = modFloat3(0.610f, 0.057f, 0.062f);
    mat.lambertianReflect = modFloat3(0.610f, 0.057f, 0.062f);
    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 2: ceiling
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, 0.f));
    trans *= glm::rotate(glm::mat4(), glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501f, 2.501f, 1.f));

    mat.lambertianReflect = modFloat3(0.730f, 0.725f, 0.729f);
    mat.color = modFloat3(0.730f, 0.725f, 0.729f);
    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 3: floor
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, -2.5f, 0.f));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501f, 2.501f, 1.f));

    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 4: back wall
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 0.f, -2.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(180.f), glm::vec3(0.f, 1.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501f, 2.501f, 1.f));

    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 5: front wall
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 0.f, 2.5f));
    trans *= glm::scale(glm::mat4(), glm::vec3(2.501f, 2.501f, 1.f));

    m_pathTracer->addShape(QUAD, trans, mat, false);

    // 6: close sphere
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(1.f, -1.5f, 0.5f));

    m_pathTracer->addShape(SPHERE, trans, mat, false);

    // 7: far sphere
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(-1.f, -1.5f, -1.25f));

    m_pathTracer->addShape(SPHERE, trans, mat, false);

    // 8: light
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.495f, 0.f));
    trans *= glm::rotate(glm::mat4(), glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3, 1.f));

    mat.color = modFloat3(0.f);
    mat.power = modFloat3(18.4f, 15.6f, 8.f) * 3.f;
    mat.emitted = mat.power / static_cast<float>(M_PI * 0.5f * 0.3f); // pi * area
    mat.lambertianReflect = modFloat3(0.f);
    mat.etaPos = 1.f;
    mat.etaNeg = 1.f;

    m_pathTracer->addAreaLight(QUAD, trans, mat, false);

    // 10: light
//    trans = glm::mat4();
//    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, -2.495f, 0.f));
//    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
//    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3, 1.f));

//    mat.power = modFloat3(18.4f, 15.6f, 8.f) * 3.f;
//    mat.emitted = mat.power / (M_PI * 0.5f * 0.3f); // pi * area
//    m_pathTracer->addAreaLight(QUAD, trans, mat, false);

//    m_pathTracer->addAreaLight(QUAD, trans, mat, false);
}



