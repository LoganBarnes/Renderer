#include <iostream>
#include "driver/Driver.hpp"
#include "world/World.hpp"
#include "RendererIOHandler.hpp"
#include "RendererConfig.hpp"



/////////////////////////////////////////////
/// \brief main
/// \return
///
/// \author Logan Barnes
/////////////////////////////////////////////
int
main(
     int          argc, ///< number of arguments
     const char **argv  ///< array of argument strings
     )
{

    shared::Driver::printProjectInfo(
                                     rndr::PROJECT_NAME,
                                     rndr::VERSION_MAJOR,
                                     rndr::VERSION_MINOR,
                                     rndr::VERSION_PATCH
                                     );

    try
    {

      //
      // create world to handle physical updates
      // and ioHandler to interface between the
      // world and the user
      //
      shared::World           world;
      rndr::RendererIOHandler io( world );

      //
      // pass world and ioHandler to driver
      // to manage update loops
      //
      shared::Driver driver( world, io );

      //
      // run program
      //
      return driver.exec( argc, argv );

    }
    catch ( std::exception &e )
    {

      std::cerr << "ERROR: program failed: " << e.what( ) << std::endl;

      return EXIT_FAILURE;

    }

    // should never reach this point
    return EXIT_FAILURE;

}
