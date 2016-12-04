#include "RendererIOHandler.hpp"

#include "glad/glad.h"
#include "graphics/glfw/GlfwWrapper.hpp"


namespace rndr
{



/////////////////////////////////////////////
/// \brief Renderer::Renderer
///
/// \author Logan Barnes
/////////////////////////////////////////////
RendererIOHandler::RendererIOHandler( shared::World &world )
  :
    OpenGLIOHandler( world, true )
{

  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

}



/////////////////////////////////////////////
/// \brief Renderer::~Renderer
///
/// \author Logan Barnes
/////////////////////////////////////////////
RendererIOHandler::~RendererIOHandler( )
{}



/////////////////////////////////////////////
/// \brief Renderer::onRender
/// \param alpha
///
/// \author Logan Barnes
/////////////////////////////////////////////
void
RendererIOHandler::onRender( const double alpha )
{

  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

} // RendererIOHandler::onRender


} // namespace rndr
