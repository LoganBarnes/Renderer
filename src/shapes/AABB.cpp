#include "AABB.hpp"


namespace shapes
{


///
/// \brief AABB::AABB
/// \param min
/// \param max
///
AABB::AABB(
           glm::vec3 min,
           glm::vec3 max
           )
  : min_( min )
  , max_( max )
{}



///
/// \brief AABB::~AABB
///
AABB::~AABB( )
{}



///
/// \brief AABB::setMin
/// \param min
///
void
AABB::setMin( const glm::vec3 min )
{

  min_ = min;

}



///
/// \brief AABB::setMax
/// \param max
///
void
AABB::setMax( const glm::vec3 max )
{

  max_ = max;
}



///
/// \brief AABB::getMin
/// \return
///
const glm::vec3&
AABB::getMin( ) const
{

  return min_;

}



///
/// \brief AABB::getMax
/// \return
///
const glm::vec3&
AABB::getMax( ) const
{

  return max_;

}



///
/// \brief AABB::intersectsAABB
/// \param ray
/// \return
///
template< typename T >
bool
AABB::intersectsAABB( const T &ray ) const
{

  float t1 = ( min_.x - ray.orig.x ) * ray.inv_dir.x;
  float t2 = ( max_.x - ray.orig.x ) * ray.inv_dir.x;

  float tmin = glm::min( t1, t2 );
  float tmax = glm::max( t1, t2 );

  t1 = ( min_.y - ray.orig.y ) * ray.inv_dir.y;
  t2 = ( max_.y - ray.orig.y ) * ray.inv_dir.y;

  tmin = glm::max( tmin, glm::min( t1, t2 ) );
  tmax = glm::min( tmax, glm::max( t1, t2 ) );

  t1 = ( min_.z - ray.orig.z ) * ray.inv_dir.z;
  t2 = ( max_.z - ray.orig.z ) * ray.inv_dir.z;

  tmin = glm::max( tmin, glm::min( t1, t2 ) );
  tmax = glm::min( tmax, glm::max( t1, t2 ) );

  return tmax > glm::max( tmin, 0.0f );

} // AABB::intersectsAABB


///
/// The two allowed ray intersect functions
///
template bool AABB::intersectsAABB< Ray3 >( const Ray3& ) const;
template bool AABB::intersectsAABB< Ray4 >( const Ray4& ) const;


} // namespace shapes
