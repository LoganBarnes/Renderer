#ifndef RendererIOHandler_hpp
#define RendererIOHandler_hpp


#include "io/OpenGLIOHandler.hpp"


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



protected:


private:

  virtual
  void onRender( const double alpha ) final;

  PathTracer *upPathTracer_;
//  std::unique_ptr< PathTracer > upPathTracer_;


};


} // namespace rndr


#endif // RendererIOHandler_hpp
