#ifndef RendererIOHandler_hpp
#define RendererIOHandler_hpp


#include <string>
#include "io/OpenGLIOHandler.hpp"


namespace graphics
{

class OpenGLWrapper;
class Camera;

}


namespace shared
{

class World;

}


namespace rndr
{

class PathTracer;


/////////////////////////////////////////////
/// \brief The RendererIOHandler class
///
/// \author Logan Barnes
/////////////////////////////////////////////
class RendererIOHandler : public shared::OpenGLIOHandler
{

public:

  ///////////////////////////////////////////////////////////////
  /// \brief Renderer
  ///////////////////////////////////////////////////////////////
  RendererIOHandler( shared::World &world );


  ///////////////////////////////////////////////////////////////
  /// \brief ~Renderer
  ///////////////////////////////////////////////////////////////
  virtual
  ~RendererIOHandler( );


  void rotateCamera (
                     double deltaX,
                     double deltaY
                     );


  void zoomCamera ( double deltaZ );


  void resetBlendTexture ( );

  void resize( int w, int h );


protected:

private:

  virtual
  void onRender ( const double alpha ) final;


  void _render (
                const std::string program,
                const std::string mainTex,
                const float       alpha,
                const bool        useTexSize  = false,
                const std::string blendTex    = ""
                );


  void _buildScene ( );



  std::unique_ptr< graphics::OpenGLWrapper > upGLWrapper_;
  std::unique_ptr< graphics::Camera > upCamera_;
  std::unique_ptr< PathTracer > upPathTracer_;

//  graphics::OpenGLWrapper *upGLWrapper_;
//  graphics::Camera *upCamera_;
//  PathTracer *upPathTracer_;


  int texWidth_;
  int texHeight_;

  int iterationsWithoutClear_;


};


} // namespace rndr


#endif // RendererIOHandler_hpp
