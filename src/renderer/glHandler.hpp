#ifndef GL_HANDLER_H
#define GL_HANDLER_H

#include <unordered_map>

typedef unsigned int GLuint;
typedef float GLfloat;

class GLHandler
{

public:
    explicit GLHandler();
    virtual ~GLHandler();

    void addProgram(const char *name, const char *vertFilePath, const char *fragFilePath);
    void setBuffer(const char *program, GLfloat *data);

    void render(const char *program);

    static std::string readFile(const char *filePath);
    static GLuint LoadShader(const char *vertex_path, const char *fragment_path);

private:
    std::unordered_map <const char*, GLuint> m_programs;
    GLuint m_vbo, m_vao;

};

#endif // GL_HANDLER_H
