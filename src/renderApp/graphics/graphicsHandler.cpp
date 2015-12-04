#define GLFW_INCLUDE_GL_3
#include <IL/il.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "graphicsHandler.hpp"


GraphicsHandler::GraphicsHandler(GLsizei width, GLsizei height)
    : m_window(NULL),
      m_viewportWidth(width),
      m_viewportHeight(height),
      m_initialized(false)
{}

GraphicsHandler::~GraphicsHandler()
{
    for (std::unordered_map <const char*, GLuint>::const_iterator it = m_programs.begin(); it != m_programs.end(); ++it)
    {
        glDeleteProgram(it->second);
    }
    for (std::unordered_map <const char*, GLuint>::const_iterator it = m_textures.begin(); it != m_textures.end(); ++it)
    {
        glDeleteTextures(1, &(it->second));
    }
    for (std::unordered_map <const char*, Buffer>::const_iterator it = m_buffers.begin(); it != m_buffers.end(); ++it)
    {
        const Buffer &buffer = it->second;
        glDeleteBuffers(1, &(buffer.vbo));
        glDeleteVertexArrays(1, &(buffer.vao));
    }
    for (std::unordered_map <const char*, Buffer>::const_iterator it = m_framebuffers.begin(); it != m_framebuffers.end(); ++it)
    {
        const Buffer &buffer = it->second;
        glDeleteFramebuffers(1, &(buffer.vbo));
        glDeleteRenderbuffers(1, &(buffer.vao));
    }

    _terminateGLFW();
}

/**
 * @brief GraphicsHandler::init
 * @param title - enter an empty title to keep the window from becoming visble
 * @param errorCallback - function called when GLFW errors occur
 * @param keyCallback - function called when keys are typed and the window is in focus
 * @return true if everything intialized correctly, false otherwise
 */
bool GraphicsHandler::init(std::string title, GLFWerrorfun errorCallback, GLFWkeyfun keyCallback)
{
    return this->_initGLFW(title, errorCallback, keyCallback) && this->_initGLEW();
}


void GraphicsHandler::addProgram(const char *name, const char *vertFilePath, const char *fragFilePath)
{
    m_programs[name] = GraphicsHandler::_loadShader(vertFilePath, fragFilePath);
}


void GraphicsHandler::addTextureArray(const char *name, GLsizei width, GLsizei height, float *array, bool linear)
{
    if (m_textures.count(name))
        glDeleteTextures(1, &(m_textures[name]));

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    if (linear)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, array);

    m_textures[name] = texture;
}


void GraphicsHandler::addTextureImage(const char *name, GLsizei width, GLsizei height, const char *)
{
    if (m_textures.count(name))
        glDeleteTextures(1, &(m_textures[name]));

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

    m_textures[name] = texture;
}


void GraphicsHandler::addUVBuffer(const char *name, const char *program, GLfloat *data, GLuint size, bool dynamic)
{
    Buffer buffer;
    if (m_buffers.count(name))
    {
        buffer = m_buffers[name];
        glDeleteBuffers(1, &(buffer.vbo));
        glDeleteVertexArrays(1, &(buffer.vao));
    }

    // Initialize the vertex buffer object.
    glGenBuffers(1, &(buffer.vbo));
    glBindBuffer(GL_ARRAY_BUFFER, (buffer.vbo));

    if (dynamic)
    {
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(GLfloat), data, GL_DYNAMIC_DRAW);
    }
    else
    {
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(GLfloat), data, GL_STATIC_DRAW);
    }

    // Initialize the vertex array object.
    glGenVertexArrays(1, &(buffer.vao));
    glBindVertexArray((buffer.vao));

    int position = glGetAttribLocation(m_programs[program], "aUV");

    glEnableVertexAttribArray(position);
    glVertexAttribPointer(
        position,
        2,                   // Num coordinates per position
        GL_FLOAT,            // Type
        GL_FALSE,            // Normalized
        sizeof(GLfloat) * 2, // Stride
        NULL                 // Array buffer offset
    );

    // Unbind buffers.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_buffers[name] = buffer;
}


void GraphicsHandler::addFramebuffer(const char *buffer, GLuint width, GLuint height, const char *texture)
{
    if (m_framebuffers.count(buffer))
    {
        Buffer buf = m_framebuffers[buffer];
        glDeleteFramebuffers(1, &(buf.vbo));
        glDeleteRenderbuffers(1, &(buf.vao));
    }

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // attach a texture to FBO color attachment point
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_textures[texture], 0);

    // attach a renderbuffer to depth attachment point
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Buffer buf;
    buf.vbo = fbo;
    buf.vao = rbo;
    m_framebuffers[buffer] = buf;
}


void GraphicsHandler::bindFramebuffer(const char *name)
{
    if (name && m_framebuffers.count(name))
        glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffers[name].vbo);
    else
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void GraphicsHandler::swapFramebuffers(const char *fbo1, const char *fbo2)
{
    Buffer temp = m_framebuffers[fbo1];
    m_framebuffers[fbo1] = m_framebuffers[fbo2];
    m_framebuffers[fbo2] = temp;
}


void GraphicsHandler::clearWindow(GLsizei width, GLsizei height)
{
    GLsizei w = m_viewportWidth;
    GLsizei h = m_viewportHeight;
    if (width > 0)
        w = width;
    if (height > 0)
        h = height;
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


void GraphicsHandler::useProgram(const char *program)
{
    glUseProgram(m_programs[program]);
}


void GraphicsHandler::setTextureUniform(const char *program, const char *uniform, const char *texture, int activeTex)
{
    switch(activeTex)
    {
    case 0:
        glActiveTexture(GL_TEXTURE0);
        break;
    case 1:
        glActiveTexture(GL_TEXTURE1);
        break;
    case 2:
        glActiveTexture(GL_TEXTURE2);
        break;
    default:
        glActiveTexture(GL_TEXTURE3);
        break;
    }
    glUniform1i(glGetUniformLocation(m_programs[program], uniform), activeTex);
    glBindTexture(GL_TEXTURE_2D, m_textures[texture]);
}


void GraphicsHandler::renderBuffer(const char* buffer, int verts, GLenum mode)
{
    glBindVertexArray(m_buffers[buffer].vao);
    glDrawArrays(mode, 0, verts);
    glBindVertexArray(0);
}


void GraphicsHandler::updateWindow()
{
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}


void GraphicsHandler::setBuffer(const char *bufferName, float *data, GLuint size)
{
    Buffer buffer = m_buffers[bufferName];
    glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
    glBindVertexArray(buffer.vao);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size * sizeof(float), data);
}


void GraphicsHandler::setBoolUniform(const char *program, const char *uniform, bool var)
{
    glUniform1i(glGetUniformLocation(m_programs[program], uniform), var);
}


void GraphicsHandler::setIntUniform(const char *program, const char *uniform, int value)
{
    glUniform1i(glGetUniformLocation(m_programs[program], uniform), value);
}


void GraphicsHandler::swapTextures(const char *tex1, const char *tex2)
{
    GLuint temp = m_textures[tex1];
    m_textures[tex1] = m_textures[tex2];
    m_textures[tex2] = temp;
}


void GraphicsHandler::setBlending(bool blend)
{
    if (blend)
    {
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
    } else
        glDisable(GL_BLEND);
}


void GraphicsHandler::setWindowShouldClose(bool close)
{
    glfwSetWindowShouldClose(m_window, static_cast<int>(close));
}


bool GraphicsHandler::checkWindowShouldClose()
{
    return glfwWindowShouldClose(m_window);
}


double GraphicsHandler::getTime()
{
    return glfwGetTime();
}


void GraphicsHandler::resize(GLsizei width, GLsizei height)
{
    m_viewportWidth = width;
    m_viewportHeight = height;
}


bool GraphicsHandler::_initGLFW(std::string title, GLFWerrorfun errorCallback, GLFWkeyfun keyCallback)
{
    if (!glfwInit())
        return false;

    std::cout << "Initialized GLFW Version: ";
    std::cout << glfwGetVersionString() << std::endl;

    m_initialized = true;

    if (errorCallback)
        glfwSetErrorCallback(errorCallback);
    else
        glfwSetErrorCallback(GraphicsHandler::_default_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (title.length() == 0)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    m_window = glfwCreateWindow(m_viewportWidth, m_viewportHeight, title.c_str(), NULL, NULL);
    if (!m_window)
        return false;

    /* Make the window's context current */
    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(1);

    if (keyCallback)
        glfwSetKeyCallback(m_window, keyCallback);
    else
        glfwSetKeyCallback(m_window, GraphicsHandler::_default_key_callback);

    return true;
}


bool GraphicsHandler::_initGLEW()
{
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
 * @brief GraphicsHandler::_terminateGLFW
 * Can be called repeatedly with no adverse effects.
 */
void GraphicsHandler::_terminateGLFW()
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


std::string GraphicsHandler::_readFile(const char *filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if(!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}


GLuint GraphicsHandler::_loadShader(const char *vertex_path, const char *fragment_path) {
    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Read shaders
    std::string vertShaderStr = _readFile(vertex_path);
    std::string fragShaderStr = _readFile(fragment_path);
    const char *vertShaderSrc = vertShaderStr.c_str();
    const char *fragShaderSrc = fragShaderStr.c_str();

    GLint result = GL_FALSE;
    int logLength;

    // Compile vertex shader
    std::cout << "Compiling vertex shader." << std::endl;
    glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
    glCompileShader(vertShader);

    // Check vertex shader
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> vertShaderError((logLength > 1) ? static_cast<unsigned long>(logLength) : 1);
    glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]);
    std::cout << &vertShaderError[0] << std::endl;

    // Compile fragment shader
    std::cout << "Compiling fragment shader." << std::endl;
    glShaderSource(fragShader, 1, &fragShaderSrc, NULL);
    glCompileShader(fragShader);

    // Check fragment shader
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> fragShaderError((logLength > 1) ? static_cast<unsigned long>(logLength) : 1);
    glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]);
    std::cout << &fragShaderError[0] << std::endl;

    std::cout << "Linking program" << std::endl;
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> programError((logLength > 1) ? static_cast<unsigned long>(logLength) : 1);
    glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
    std::cout << &programError[0] << std::endl;

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}


void GraphicsHandler::_default_key_callback(GLFWwindow* window, int key, int, int action, int)
{
    std::cout << "key callback" << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void GraphicsHandler::_default_error_callback(int error, const char* description)
{
    std::cerr << "ERROR: (" << error << ") " << description << std::endl;
}


