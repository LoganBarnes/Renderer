#ifndef Renderer_src_kdTree_KDRenderer_hpp
#define Renderer_src_kdTree_KDRenderer_hpp


#include "AABB.hpp"


namespace graphics
{

class OpenGLWrapper;

}


namespace kdtree
{


class KDRenderer : public shapes::AABB
{

public:

  KDRenderer( graphics::OpenGLWrapper &graphics );

  ~KDRenderer( );

  void render( graphics::OpenGLWrapper &graphics ) const;


private:

protected:

private:


};


} // namespace kdtree


#endif // Renderer_src_kdTree_KDRenderer_hpp
