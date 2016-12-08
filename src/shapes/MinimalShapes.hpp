#ifndef Renderer_src_shapes_MinimalShapes_hpp
#define Renderer_src_shapes_MinimalShapes_hpp


#include "glm/glm.hpp"


namespace shapes
{

///
/// \brief The Ray3 struct
///
struct Ray3
{

  glm::vec3 orig;
  glm::vec3 dir;
  glm::vec3 inv_dir;

  Ray3(
       glm::vec3 orig_ = glm::vec3( 0.0f, 0.0f, 0.0f ),
       glm::vec3 dir_  = glm::vec3( 0.0f, 0.0f, 1.0f )
       )
    : orig   ( orig_ )
    , dir    ( dir_ )
    , inv_dir( 1.0f / dir )
  {}

};


///
/// \brief The Ray4 struct
///
struct Ray4
{

  glm::vec4 orig;
  glm::vec4 dir;
  glm::vec4 inv_dir;

  Ray4(
       glm::vec4 orig_ = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
       glm::vec4 dir_  = glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f )
       )
    : orig   ( orig_ )
    , dir    ( dir_ )
    , inv_dir( 1.0f / dir )
  {}

};


} // namespace shapes


#endif // Renderer_src_shapes_MinimalShapes_hpp
