#ifndef RendererIOHandler_hpp
#define RendererIOHandler_hpp


#include "io/OpenGLIOHandler.hpp"


namespace shared
{

class World;

}


namespace rndr
{


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

  // RendererWorld &world_;

};


} // namespace rndr


#endif // RendererIOHandler_hpp
