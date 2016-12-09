#include "KDRenderer.hpp"
#include "graphics/opengl/OpenGLWrapper.hpp"


namespace kdtree
{


KDRenderer::KDRenderer( graphics::OpenGLWrapper &/*graphics*/ )
  : shapes::AABB( )
{

}

KDRenderer::~KDRenderer( )
{

}

void
KDRenderer::render( graphics::OpenGLWrapper &/*graphics*/ ) const
{

}


} // namespace kdtree


