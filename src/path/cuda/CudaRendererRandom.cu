#include <curand.h>
#include <curand_kernel.h>
#include "RenderObjects.hpp"


extern "C"
{

  /**
   * @brief randHemi
   * @param normal
   * @param state
   * @param id
   * @return
   */
  __device__
  float3
  randHemi(
           float3       normal,
           curandState *state,
           int          id
           )
  {
    float3 rand;

    rand.x = curand_uniform( state + id ) * 1.999999f - 1.f;
    rand.y = curand_uniform( state + id ) * 1.999999f - 1.f;
    rand.z = curand_uniform( state + id ) * 1.999999f - 1.f;

    if ( dot( normal, rand ) < 0.f )
    {
      rand = -rand;
    }

    return rand;
  }



  /**
   * @brief kernel_randCosHemi
   * @param state
   * @param idx
   * @return
   */
  __device__
  float3
  randCosHemi(
              float3       normal,
              curandState *state,
              int          id
              )
  {
    const float e1 = 1.f - curand_uniform( state + id );
    const float e2 = 1.f - curand_uniform( state + id );

    // Jensen's method
    const float sin_theta = sqrtf( 1.0f - e1 );
    const float cos_theta = sqrtf( e1 );
    const float phi       = 6.28318531f * e2;

    float3 rand = make_float3( cos( phi ) * sin_theta,
                              sin( phi ) * sin_theta,
                              cos_theta );

    // Make a coordinate system
    const float3 &Z = normal;

    float3 X, Y;

    // GET TANGENTS
    X = ( abs( normal.x ) < 0.9f ) ? make_float3( 1.f, 0.f, 0.f ) : make_float3( 0.f, 1.f, 0.f );

    // Remove the part that is parallel to Z
    X -= normal * dot( normal, X );
    X /= length( X );   // normalize

    Y = cross( normal, X );

    return rand.x * X
           + rand.y * Y
           + rand.z * Z;

  } // randCosHemi



  /**
   * @brief samplePoint
   * @param state
   * @param id
   * @param s
   * @return
   */
  __device__
  SurfaceElement
  samplePoint(
              curandState *state,
              int          id,
              Shape        shape
              )
  {
    SurfaceElement surfel;

    if ( shape.type == QUAD )
    {
      float x = 1.f - curand_uniform( state + id );
      float y = 1.f - curand_uniform( state + id );

      x = x * 2.f - 1.f;
      y = y * 2.f - 1.f;

//      surfel.point = make_float3( shape.trans * make_float4( 0.f, 0.f, 0.f, 1.f ) );
      surfel.point    = make_float3( shape.trans * make_float4( x, y, 0.f, 1.f ) );
      surfel.normal   = normalize( shape.normInv * make_float3( 0.f, 0.f, -1.f ) );
      surfel.material = shape.material;
      surfel.index    = static_cast< int >( shape.index );
    }
    else
    {
      surfel.index = -1;
    }

    return surfel;
  } // samplePoint



}
