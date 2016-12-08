#ifndef Renderer_src_kdTree_KDNode_hpp
#define Renderer_src_kdTree_KDNode_hpp


#include <vector>


#include "AABB.hpp"


namespace kdtree
{

template< typename T >
class KDElement;


template< typename T >
class KDNode : public shapes::AABB
{

public:

  KDNode( );

  ~KDNode( );

  // setters
  void setLeftChild  ( KDNode< T > *pLeft );
  void setRightChild ( KDNode< T > *pRight );
  void setElements   ( std::vector< KDElement< T >* > elements );

  // getters
  KDNode< T > *getLeftChild ( );
  KDNode< T > *getRightChild ( );

  std::vector< KDElement< T >* > &getElements ( );

  // return true if node has appropriate child
  bool hasLeftChild ( );
  bool hasRightChild ( );

  // void displayNode( KDRenderer *pRenderer );


private:

  // children nodes
  KDNode *pLeft_;
  KDNode *pRight_;

  // depth of node
  int depth_;

  // list of elements (empty if not a leaf node)
  std::vector< KDElement< T >* > elements_;

};



template< typename T >
KDNode< T >::KDNode( )
  : pLeft_ ( nullptr )
  , pRight_( nullptr )
  , depth_ ( -1 ) // means node is invalid
{}



template< typename T >
KDNode< T >::~KDNode( )
{

  if ( pLeft_ )
  {

    delete pLeft_;
    pLeft_ = nullptr;

  }

  if ( pRight_ )
  {

    delete pRight_;
    pRight_ = nullptr;

  }

  //
  // element deletion is handled elsewhere because
  // multiple nodes can point to the same elements
  //

}



template< typename T >
void
KDNode< T >::setLeftChild( KDNode< T > *pLeft )
{

  pLeft_ = pLeft;

}



template< typename T >
void
KDNode< T >::setRightChild( KDNode< T > *pRight )
{

  pRight_ = pRight;

}



template< typename T >
bool
KDNode< T >::hasLeftChild( )
{

  return pLeft_ != NULL;

}



template< typename T >
bool
KDNode< T >::hasRightChild( )
{

  return pRight_ != NULL;

}



template< typename T >
KDNode< T >*
KDNode< T >::getLeftChild( )
{

  return pLeft_;

}



template< typename T >
KDNode< T >*
KDNode< T >::getRightChild( )
{

  return pRight_;

}



template< typename T >
void
KDNode< T >::setElements( std::vector< KDElement< T >* > elements )
{

  elements_ = elements;

}



template< typename T >
std::vector< KDElement< T >* >&
KDNode< T >::getElements( )
{

  return elements_;

}



} // namespace kdtree


#endif // Renderer_src_kdTree_KDNode_hpp
