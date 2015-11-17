#ifndef APPLICATION_H
#define APPLICATION_H

typedef void (* GLFWerrorfun)(int,const char*);
typedef struct GLFWwindow GLFWwindow;

class Application
{

public:
    virtual ~Application() {}

    virtual void init() {}

    virtual void update(double) {}
    virtual void handleKeyInput() {}

    virtual void render(int /*interp = 1.0*/) {}
};

#endif // APPLICATION_H
