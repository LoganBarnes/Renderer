#ifndef RendererCallback_hpp
#define RendererCallback_hpp

#include "io/SharedCallback.hpp"

namespace rndr
{

class RendererIOHandler;


class RendererCallback : public shared::SharedCallback
{
public:

  RendererCallback( RendererIOHandler &handler );

  virtual
  ~RendererCallback( );


  virtual
  void handleWindowSize (
                         GLFWwindow *pWindow,
                         int         width,
                         int         height
                         );

  virtual
  void handleMouseButton (
                          GLFWwindow *pWindow,
                          int         button,
                          int         action,
                          int         mods
                          );

  virtual
  void handleKey (
                  GLFWwindow *pWindow,
                  int         key,
                  int         scancode,
                  int         action,
                  int         mods
                  );

  virtual
  void handleCursorPosition (
                             GLFWwindow *pWindow,
                             double      xpos,
                             double      ypos
                             );

  virtual
  void handleScroll (
                     GLFWwindow *pWindow,
                     double      xoffset,
                     double      yoffset
                     );


private:

  RendererIOHandler &handler_;

  bool leftMouseDown_;
  bool rightMouseDown_;

  bool shiftDown_;

  double prevX_;
  double prevY_;

};


} //  namespace rndr


#endif // RendererCallback_hpp
