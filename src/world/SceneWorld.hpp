#ifndef Renderer_src_world_SceneWorld_hpp
#define Renderer_src_world_SceneWorld_hpp


#include <vector>
#include "world/World.hpp"
#include "ShapeInterface.hpp"


namespace kdtree
{

template< typename T >
class KDTree;

}


namespace rndr
{

/////////////////////////////////////////////
/// \brief The SceneWorld class
///
/// \author Logan Barnes
/////////////////////////////////////////////
class SceneWorld : public shared::World
{

public:

  ///////////////////////////////////////////////////////////////
  /// \brief SceneWorld
  ///////////////////////////////////////////////////////////////
  SceneWorld( );


  ///////////////////////////////////////////////////////////////
  /// \brief ~SceneWorld
  ///////////////////////////////////////////////////////////////
  ~SceneWorld( );



private:

  void _buildScene( );


  std::vector< shapes::ShapeInterface > shapes_;

  // std::unique_ptr< KDTree > upKDTree_;
  kdtree::KDTree< shapes::ShapeInterface > *pKDTree_;


};


} // namespace rndr


#endif // Renderer_src_world_SceneWorld_hpp
