#ifndef GRAPHICS_HANDLER_H
#define GRAPHICS_HANDLER_H

//#include <GL/glew.h>
#include <unordered_map>

typedef int GLsizei;
typedef unsigned int GLuint;
typedef float GLfloat;
typedef uint32_t GLenum;

typedef struct GLFWwindow GLFWwindow;

typedef void (* GLFWerrorfun)(int,const char*);
typedef void (* GLFWkeyfun)(GLFWwindow*,int,int,int,int);

struct Buffer
{
    GLuint vbo;
    GLuint vao;
};


class GraphicsHandler
{

public:
    explicit GraphicsHandler(GLsizei width = 640, GLsizei height = 480);
    virtual ~GraphicsHandler();

    bool init(std::string title = "Window", GLFWerrorfun errorCallback = NULL, GLFWkeyfun keyCallback = NULL);

    // getters
    GLuint getTexture(const char *name) { return m_textures[name]; }
    GLsizei getViewportWidth() { return m_viewportWidth; }
    GLsizei getViewportHeight() { return m_viewportHeight; }


    void addProgram(const char *name, const char *vertFilePath, const char *fragFilePath);
    void addTextureArray(const char *name, GLsizei width, GLsizei height, float *array = NULL);
    void addTextureImage(const char *name, GLsizei width, GLsizei height, const char *filename);

    void addUVBuffer(const char *buffer, const char *program, GLfloat *data, GLuint size, bool dynamic = false);

    void render(const char *program, const char *buffer, int verts, GLenum mode, const char *texture);
    void clearWindow();
    void useProgram(const char *program);
    void setTexture(const char *program, const char *texture);
    void renderBuffer(const char *buffer, int verts, GLenum mode);
    void updateWindow();

    void setBoolUniform(const char *program, const char *uniform, bool var);
    void setBuffer(const char *bufferName, float *data, GLuint size);

    bool checkWindowShouldClose();
    void resize(GLsizei width, GLsizei height);


private:
    bool _initGLFW(std::__1::string title, GLFWerrorfun errorCallback, GLFWkeyfun keyCallback);
    bool _initGLEW();

    void _terminateGLFW();

    static std::string _readFile(const char *filePath);
    static GLuint _loadShader(const char *vertex_path, const char *fragment_path);

    static void _default_key_callback(GLFWwindow* window, int key, int, int action, int);
    static void _default_error_callback(int error, const char* description);


    GLFWwindow *m_window;

    std::unordered_map <const char*, GLuint> m_programs;
    std::unordered_map <const char*, GLuint> m_textures;
    std::unordered_map <const char*, Buffer> m_buffers;

    GLsizei m_viewportWidth, m_viewportHeight;


    bool m_initialized;

};

#endif // GRAPHICS_HANDLER_H