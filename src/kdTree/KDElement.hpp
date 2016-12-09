#ifndef Renderer_src_kdTree_KDElement_hpp
#define Renderer_src_kdTree_KDElement_hpp


#include "glm/glm.hpp"


namespace kdtree
{


template< typename T >
class KDElement
{

public:

  KDElement(
            const glm::vec3 &min,
            const glm::vec3 &max,
            const glm::vec3 &pos,
            const T         *pElement
            );

  ~KDElement( );

  // getters
  const glm::vec3 &getMin ( ) const;
  const glm::vec3 &getMax ( ) const;
  const glm::vec3 &getPos ( ) const;

  const T &getElement ( ) const;


private:

  // minimum, maximum, and center coordinates
  const glm::vec3 min_;
  const glm::vec3 max_;
  const glm::vec3 pos_;

  // value in kdtree
  const T *pElement_;

};



template< typename T >
KDElement< T >::KDElement(
                          const glm::vec3 &min,
                          const glm::vec3 &max,
                          const glm::vec3 &pos,
                          const T         *pElement
                          )
  : min_     ( min )
  , max_     ( max )
  , pos_     ( pos )
  , pElement_( pElement )
{}


template< typename T >
KDElement< T >::~KDElement( )
{}



template< typename T >
const glm::vec3&
KDElement< T >::getMin( ) const
{

  return min_;

}



template< typename T >
const glm::vec3&
KDElement< T >::getMax( ) const
{

  return max_;

}



template< typename T >
const glm::vec3&
KDElement< T >::getPos( ) const
{

  return pos_;

}



template< typename T >
const T&
KDElement< T >::getElement( ) const
{

  return *pElement_;

}



} // namespace kdtree


#endif // Renderer_src_kdTree_KDElement_hpp
