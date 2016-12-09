#ifndef Renderer_src_shapes_AABB_hpp
#define Renderer_src_shapes_AABB_hpp


#include "glm/glm.hpp"
#include "MinimalShapes.hpp"


namespace shapes
{


class AABB
{

public:

  AABB(
       glm::vec3 min = glm::vec3( -1 ),
       glm::vec3 max = glm::vec3(  1 )
       );

  ~AABB( );

  //
  // sets the minimum and maximum points defining the box
  //
  void setMin ( const glm::vec3 min );
  void setMax ( const glm::vec3 max );

  //
  // returns the minimum and maximum point defining the box
  //
  const glm::vec3 &getMin ( ) const;
  const glm::vec3 &getMax ( ) const;

  //
  // return true if the given ray intersects the box
  //
  template< typename T >
  bool intersectsAABB ( const T &ray ) const;


private:

  //
  // two corner points defining the cube
  //
  glm::vec3 min_, max_;

};


} // namespace shapes



#endif // Renderer_src_shapes_AABB_hpp
