#ifndef Renderer_src_kdTree_KDTree_hpp
#define Renderer_src_kdTree_KDTree_hpp


#include <vector>
#include "glm/glm.hpp"

#include "KDNode.hpp"


namespace kdtree
{

template< typename T >
class KDElement;

class KDRenderer;


template< typename T >
class KDTree
{

public:

  KDTree(
         const std::vector< KDElement< T >* > &elements,
         glm::vec3                             min,
         glm::vec3                             max
         );

  ~KDTree( );


  void displayTree( KDRenderer *pRenderer );


private:

  void _buildTree (
                   int                                   depth,
                   const std::vector< KDElement< T >* > &elements,
                   KDNode< T >                          *pNode
                   );


  // The root of the tree
  // std::unique_ptr< KDNode > upRoot_;
  KDNode< T > *pRoot_;

};



template< typename T >
KDTree< T >::KDTree(
                    const std::vector< KDElement< T >* > &elements,
                    glm::vec3                             min,
                    glm::vec3                             max
                    )
  : pRoot_( new KDNode< T >( ) )
{

  pRoot_->setMin( min );
  pRoot_->setMax( max );
  _buildTree( 0, elements, pRoot_ );

}



template< typename T >
KDTree< T >::~KDTree( )
{

  delete pRoot_;

}


template< typename T >
void
KDTree< T >::displayTree( KDRenderer *pRenderer )
{

  pRoot_->displayNode( pRenderer );

}



template< typename T >
void
KDTree< T >::_buildTree(
                        int                                   /*depth*/,
                        const std::vector< KDElement< T >* >& /*elements*/,
                        KDNode< T >*                          /*pNode*/
                        )
{}



} // namespace kdtree


#endif // Renderer_src_kdTree_KDTree_hpp
