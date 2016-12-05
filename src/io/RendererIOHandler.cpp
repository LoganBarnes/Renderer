#include "RendererIOHandler.hpp"

#include <vector>

#include "glad/glad.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "graphics/glfw/GlfwWrapper.hpp"
#include "graphics/opengl/OpenGLWrapper.hpp"
#include "graphics/Camera.hpp"
#include "PathTracer.hpp"
#include "RenderObjects.hpp"
#include "RendererCallback.hpp"


namespace rndr
{


namespace
{

constexpr int defaultWidth  = 512;
constexpr int defaultHeight = 512;

constexpr bool antialias = true;

constexpr float lightScaling = 1.0f;



#ifdef USE_CUDA

float3
modFloat3( float x ) { return make_float3( x ); }

float3
modFloat3(
          float x,
          float y,
          float z
          ) { return make_float3( x, y, z ); }

#else

glm::vec3
modFloat3( float x ) { return glm::vec3( x ); }

glm::vec3
modFloat3(
          float x,
          float y,
          float z
          ) { return glm::vec3( x, y, z ); }

#endif // ifdef USE_CUDA


}


/////////////////////////////////////////////
/// \brief Renderer::Renderer
///
/// \author Logan Barnes
/////////////////////////////////////////////
RendererIOHandler::RendererIOHandler( shared::World &world )
  : OpenGLIOHandler( world, true, defaultWidth, defaultHeight, false )
  , upGLWrapper_   ( new graphics::OpenGLWrapper( defaultWidth, defaultHeight ) )
  , upCamera_      ( new graphics::Camera( ) )
  , upPathTracer_  ( new PathTracer( ) )
  , texWidth_      ( defaultWidth )
  , texHeight_     ( defaultHeight )
  , iterationsWithoutClear_( 1 )
{

  std::unique_ptr< shared::SharedCallback > upCallback( new RendererCallback( *this ) );

  upGlfwWrapper_->setCallback( std::move( upCallback ) );

  if ( antialias )
  {

    texWidth_  *= 2;
    texHeight_ *= 2;

  }

  //
  // OpenGL
  //

  upGLWrapper_->init( );

  upGLWrapper_->setViewportSize( defaultWidth, defaultHeight );

  std::string vertShader = SHADER_PATH + "screenSpace/shader.vert";
  std::string fragShader = SHADER_PATH + "blending/shader.frag";

  upGLWrapper_->addProgram(
                           "path",
                           vertShader.c_str( ),
                           fragShader.c_str( )
                           );

  // temporary
  std::vector< float > vbo =
  {
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
    1.0, -1.0, -1.0,
    1.0,  1.0, -1.0
  };

  upGLWrapper_->addUVBuffer(
                            "screenBuffer",
                            "path",
                            vbo.data( ),
                            vbo.size( )
                            );

  std::vector< float > data( static_cast< size_t >( texWidth_ * texHeight_ * 4 ) );


  upGLWrapper_->addTextureArray( "currTex",   texWidth_, texHeight_ );
  upGLWrapper_->addTextureArray( "blendTex1", texWidth_, texHeight_, nullptr, true );
  upGLWrapper_->addTextureArray( "blendTex2", texWidth_, texHeight_, nullptr, true );
  upGLWrapper_->addFramebuffer( "framebuffer1", texWidth_, texHeight_, "blendTex1" );
  upGLWrapper_->addFramebuffer( "framebuffer2", texWidth_, texHeight_, "blendTex2" );


  //
  // camera
  //

  upCamera_->setAspectRatio( defaultWidth * 1.0f / defaultHeight );
  upCamera_->updateOrbit( 7.0f, 0.0f, 0.0f );


  //
  // path tracer
  //

  upPathTracer_->init(
                      static_cast< unsigned >( texWidth_ ),
                      static_cast< unsigned >( texHeight_ )
                      );

  upPathTracer_->register2DTexture( "currTex", upGLWrapper_->getTexture( "currTex" ) );
  upPathTracer_->setScaleViewInvEye( upCamera_->getEye( ), upCamera_->getScaleViewInvMatrix( ) );

  _buildScene( );

  upPathTracer_->updateShapesOnGPU( );

  resetBlendTexture( );

}



/////////////////////////////////////////////
/// \brief Renderer::~Renderer
///
/// \author Logan Barnes
/////////////////////////////////////////////
RendererIOHandler::~RendererIOHandler( )
{

  upPathTracer_->unregisterTexture( "currTex" );

  delete upGLWrapper_;
  delete upCamera_;
  delete upPathTracer_;
}



void
RendererIOHandler::rotateCamera(
                                double deltaX,
                                double deltaY
                                )
{

  upCamera_->updateOrbit( 0.f, static_cast< float >( deltaX ), static_cast< float >( deltaY ) );
  upPathTracer_->setScaleViewInvEye( upCamera_->getEye( ), upCamera_->getScaleViewInvMatrix( ) );

  resetBlendTexture( );

}



void
RendererIOHandler::zoomCamera( double deltaZ )
{

  upCamera_->updateOrbit( static_cast< float >( deltaZ * 0.25 ), 0.f, 0.f );
  upPathTracer_->setScaleViewInvEye( upCamera_->getEye( ), upCamera_->getScaleViewInvMatrix( ) );

  resetBlendTexture( );

}



void
RendererIOHandler::resetBlendTexture( )
{

  // clear blending texture
  upGLWrapper_->bindFramebuffer( "framebuffer1" );
  upGLWrapper_->clearWindow( texWidth_, texHeight_ );

  // reset iteration number
  iterationsWithoutClear_ = 1;

}



void
RendererIOHandler::resize(
                          int w,
                          int h
                          )
{

  upGLWrapper_->setViewportSize( w, h );

  upCamera_->setAspectRatio( w * 1.0f / h );
  upPathTracer_->setScaleViewInvEye( upCamera_->getEye( ), upCamera_->getScaleViewInvMatrix( ) );

  resetBlendTexture( );

}



/////////////////////////////////////////////
/// \brief Renderer::onRender
/// \param alpha
///
/// \author Logan Barnes
/////////////////////////////////////////////
void
RendererIOHandler::onRender( const double )
{

  upPathTracer_->tracePath(
                           "currTex",
                           static_cast< uint >( texWidth_ ),
                           static_cast< uint >( texHeight_ ),
                           lightScaling
                           );

  // blend to texture
  upGLWrapper_->bindFramebuffer( "framebuffer2" );
  _render( "path", "currTex", 1.0f / iterationsWithoutClear_, true, "blendTex1" );

  // render texture
  upGLWrapper_->bindFramebuffer( "" );
  _render( "path", "blendTex2", 1.0, false );

  // swap texture and framebuffer values so we can use the same names
  upGLWrapper_->swapTextures( "blendTex1", "blendTex2" );
  upGLWrapper_->swapFramebuffers( "framebuffer1", "framebuffer2" );

  ++iterationsWithoutClear_;


} // RendererIOHandler::onRender



void
RendererIOHandler::_render(
                           const std::string program,
                           const std::string mainTex,
                           const float       alpha,
                           const bool        useTexSize,
                           const std::string blendTex
                           )

{

  glm::vec2 screenSize;

  if ( useTexSize )
  {

    upGLWrapper_->clearWindow( texWidth_, texHeight_ );
    screenSize = glm::vec2( texWidth_, texHeight_ );

  }
  else
  {

    screenSize = glm::vec2(
                           upGLWrapper_->getViewportWidth( ),
                           upGLWrapper_->getViewportHeight( )
                           );

    upGLWrapper_->clearWindow( );

  }

  upGLWrapper_->useProgram( program );
  upGLWrapper_->setTextureUniform( program, "texture1", mainTex, 0 );

  upGLWrapper_->setFloatUniform( program, "texSize", glm::value_ptr( screenSize ), 2 );

  if ( blendTex.length( ) > 0 )
  {

    upGLWrapper_->setTextureUniform( program, "texture2", blendTex, 1 );

  }

  upGLWrapper_->setFloatUniform( program, "alpha", &alpha );
  upGLWrapper_->renderBuffer( "screenBuffer", 4, GL_TRIANGLE_STRIP );

} // _render



void
RendererIOHandler::_buildScene( )
{
  // 0: left (green) wall
  glm::mat4 trans = glm::mat4( );

  trans *= glm::translate( glm::mat4( ), glm::vec3( -2.5f, 0.f, 0.f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( -90.f ), glm::vec3( 0.f, 1.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.5f, 2.5f, 1.f ) );

  Material mat;
  mat.power             = modFloat3( 0.f );
  mat.emitted           = modFloat3( 0.f );
  mat.color             = modFloat3( 0.117f, 0.472f, 0.115f );
  mat.lambertianReflect = modFloat3( 0.117f, 0.472f, 0.115f );
  mat.etaPos            = 1.f;
  mat.etaNeg            = 1.f;
  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 1: right (red) wall
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 2.5f, 0.f, 0.f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( 90.f ), glm::vec3( 0.f, 1.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.5f, 2.5f, 1.f ) );

  mat.color             = modFloat3( 0.610f, 0.057f, 0.062f );
  mat.lambertianReflect = modFloat3( 0.610f, 0.057f, 0.062f );
  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 2: ceiling
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 0.f, 2.5f, 0.f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( -90.f ), glm::vec3( 1.f, 0.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.501f, 2.501f, 1.f ) );

  mat.lambertianReflect = modFloat3( 0.730f, 0.725f, 0.729f );
  mat.color             = modFloat3( 0.730f, 0.725f, 0.729f );
  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 3: floor
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 0.f, -2.5f, 0.f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( 90.f ), glm::vec3( 1.f, 0.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.501f, 2.501f, 1.f ) );

  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 4: back wall
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 0.f, 0.f, -2.5f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( 180.f ), glm::vec3( 0.f, 1.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.501f, 2.501f, 1.f ) );

  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 5: front wall
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 0.f, 0.f, 2.5f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 2.501f, 2.501f, 1.f ) );

  upPathTracer_->addShape( QUAD, trans, mat, false );

  // 6: close sphere
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 1.f, -1.5f, 0.5f ) );

  upPathTracer_->addShape( SPHERE, trans, mat, false );

  // 7: far sphere
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( -1.f, -1.5f, -1.25f ) );

  upPathTracer_->addShape( SPHERE, trans, mat, false );

  // 8: light
  trans  = glm::mat4( );
  trans *= glm::translate( glm::mat4( ), glm::vec3( 0.f, 2.495f, 0.f ) );
  trans *= glm::rotate( glm::mat4( ), glm::radians( -90.f ), glm::vec3( 1.f, 0.f, 0.f ) );
  trans *= glm::scale( glm::mat4( ), glm::vec3( 0.5f, 0.3, 1.f ) );

  mat.color             = modFloat3( 0.f );
  mat.power             = modFloat3( 18.4f, 15.6f, 8.f ) * 3.f;
  mat.emitted           = mat.power / static_cast< float >( M_PI * 0.5f * 0.3f ); // pi * area
  mat.lambertianReflect = modFloat3( 0.f );
  mat.etaPos            = 1.f;
  mat.etaNeg            = 1.f;

  upPathTracer_->addAreaLight( QUAD, trans, mat, false );

  // 10: light
//    trans = glm::mat4();
//    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, -2.495f, 0.f));
//    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
//    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3, 1.f));

//    mat.power = modFloat3(18.4f, 15.6f, 8.f) * 3.f;
//    mat.emitted = mat.power / (M_PI * 0.5f * 0.3f); // pi * area
//    upPathTracer_->addAreaLight(QUAD, trans, mat, false);

//    upPathTracer_->addAreaLight(QUAD, trans, mat, false);
} // _buildScene



} // namespace rndr
