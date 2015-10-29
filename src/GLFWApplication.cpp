#define GLFW_INCLUDE_GL_3
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "GLFWApplication.hpp"
#include "Application.hpp"


GLFWApplication::GLFWApplication()
    : m_app(NULL),
      m_window(NULL),
      m_width(640),
      m_height(480),
      m_fps(60.0),
      m_initialized(false)
{}


GLFWApplication::~GLFWApplication()
{
    if (m_app)
        delete m_app;

    std::cout << "terminating GLFW" << std::endl;
    this->_terminateGLFW();
}


void GLFWApplication::setInternalApplication(Application *app)
{
    if (m_app)
        delete m_app;

    m_app = app;

    if (m_initialized)
        m_app->init();
}


void GLFWApplication::setWindowSize(int width, int height)
{
    m_width = width;
    m_height = height;
}


int GLFWApplication::execute(int argc, const char **argv, const char *title)
{

    /* Initialize the library */
    if (!this->_initGLFW(title))
        return 1;

    m_fps = 60.0;

    /*
     * physics game loop (not necessary for this but cool).
     * http://gafferongames.com/game-physics/fix-your-timestep/
     */
    double maxTimeStep = 0.25;

    double totalTime = 0.0;
    double deltaTime = 1.0 / m_fps;

    double currentTime = glfwGetTime();
    double accumulator = 0.0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(m_window))
    {

        double newTime = glfwGetTime();
        double frameTime = newTime - currentTime;

        if (frameTime > maxTimeStep)
            frameTime = maxTimeStep;

        currentTime = newTime;

        accumulator += frameTime;

        while ( accumulator >= deltaTime )
        {
            /* update */
            if (m_app)
                m_app->update(deltaTime);
//            previousState = currentState;
//            integrate( currentState, t, deltaTime );
            totalTime += deltaTime;
            accumulator -= deltaTime;
        }

        const double alpha = accumulator / deltaTime;

//        State state = currentState * alpha +
//            previousState * ( 1.0 - alpha );
        if (m_app)
            m_app->render(alpha);

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }

    // GLFW is terminated in the destructor
    return 0;
}


bool GLFWApplication::_initGLFW(const char* title)
{
    if (!glfwInit())
        return false;

    std::cout << "Initialized GLFW Version: ";
    std::cout << glfwGetVersionString() << std::endl;

    m_initialized = true;
    glfwSetErrorCallback(GLFWApplication::error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(m_width, m_height, title, NULL, NULL);
    if (!m_window)
        return false;

    /* Make the window's context current */
    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(1);
    glfwSetKeyCallback(m_window, GLFWApplication::key_callback);

    /*
     * Set OpenGL properties
     */
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    glGetError(); // Clear errors after call to glewInit
    if (GLEW_OK != err)
    {
        // Problem: glewInit failed, something is seriously wrong.
        std::cerr << "Error initializing glew: ";
        std::cerr << glewGetErrorString(err) << std::endl;
        return false;
    }

    if (m_app)
        m_app->init();

    // Enable depth testing, so that objects are occluded based on depth instead of drawing order.
    glEnable(GL_DEPTH_TEST);

    // Move the polygons back a bit so lines are still drawn even though they are coplanar with the
    // polygons they came from, which will be drawn before them.
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1, -1);

    // Enable back-face culling, meaning only the front side of every face is rendered.
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // Specify that the front face is represented by vertices in counterclockwise order (this is
    // the default).
    glFrontFace(GL_CCW);

    // Specify the color used when glClear is called
    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f);

    return true;
}

/**
 * @brief GLFWApplication::_terminateGLFW
 * Can be called repeatedly with no adverse effects.
 */
void GLFWApplication::_terminateGLFW()
{
    if (!m_initialized)
        return;

    if (m_window)
    {
        glfwDestroyWindow(m_window);
        m_window = NULL;
    }

    glfwTerminate();

    m_initialized = false;
}


void GLFWApplication::key_callback(GLFWwindow* window, int key, int, int action, int)
{
    std::cout << "key callback" << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void GLFWApplication::error_callback(int error, const char* description)
{
    std::cerr << "ERROR: (" << error << ") " << description << std::endl;
}



