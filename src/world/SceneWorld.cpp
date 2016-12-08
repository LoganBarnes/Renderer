#include "SceneWorld.hpp"
#include "KDTree.hpp"


namespace rndr
{


/////////////////////////////////////////////
/// \brief SceneWorld::SceneWorld
///
/// \author Logan Barnes
/////////////////////////////////////////////
SceneWorld::SceneWorld( )
  : shared::World( )
  , pKDTree_( nullptr )
{

  _buildScene( );

}


/////////////////////////////////////////////
/// \brief SceneWorld::~SceneWorld
///
/// \author Logan Barnes
/////////////////////////////////////////////
SceneWorld::~SceneWorld( )
{

  if ( pKDTree_ )
  {

    delete pKDTree_;
    pKDTree_ = nullptr;

  }

}



/////////////////////////////////////////////
/// \brief SceneWorld::_buildScene
///
/// \author Logan Barnes
/////////////////////////////////////////////
void
SceneWorld::_buildScene( )
{

  ///\todo: add shapes to vector then build tree

}



} // namespace rndr
