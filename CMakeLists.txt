cmake_minimum_required (VERSION 2.6)
project (Renderer)

# The version number.
set (Renderer_VERSION_MAJOR 1)
set (Renderer_VERSION_MINOR 0)

# should we use cuda functions?
option(USE_CUDA "Use CUDA kernels if available" ON)

# should we use graphics functions?
option(USE_GRAPHICS "Use CUDA kernels if available" ON)

# resource location
set(RESOURCES_PATH "${PROJECT_SOURCE_DIR}/res")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
        "${PROJECT_SOURCE_DIR}/src/renderer-config.hpp.in"
        "${PROJECT_BINARY_DIR}/src/renderer-config.hpp"
)

# add the binary tree to the search path for include files
# so that we will find renderer-config.hpp
include_directories (
        "${PROJECT_BINARY_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src/application"
        "${PROJECT_SOURCE_DIR}/src/shared"
        "${PROJECT_SOURCE_DIR}/src/renderApp"
)

add_subdirectory("${PROJECT_SOURCE_DIR}/src/application")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/shared")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/renderApp")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/renderApp/graphics")

# resources
add_subdirectory("${RESOURCES_PATH}/shaders")

## add glm functionality
#find_package(GLM REQUIRED)
#if (GLM_FOUND)
#    message("-- Found GLM: ${GLM_INCLUDE_DIRS}")
#    include_directories("${GLM_INCLUDE_DIRS}")
#endif (GLM_FOUND)

# using the above method ^ adds the /usr/local/include dir
# to the qt creator build tree (which is annoying)
include_directories("/usr/local/include/glm")

if (USE_GRAPHICS)
include_directories("${PROJECT_SOURCE_DIR}/src/renderApp/graphics")
# add opengl functionality
find_package(OpenGL REQUIRED)
if (OPENGL_FOUND)
    message("-- Found OpenGL: ${OPENGL_INCLUDE_DIR}")
    include_directories(${OPENGL_INCLUDE_DIR})
    set (EXTRA_LIBS ${EXTRA_LIBS} ${OPENGL_LIBRARIES})
endif (OPENGL_FOUND)

# add glew functionality
find_package(GLEW REQUIRED)
if (GLEW_FOUND) # GLEW has it's own 'found' message
    include_directories(${GLEW_INCLUDE_DIRS})
    set (EXTRA_LIBS ${EXTRA_LIBS} ${GLEW_LIBRARIES})
endif (GLEW_FOUND)

# add glfw3 functionality
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
if (GLFW_FOUND)
    message("-- Found GLFW: ${GLFW_INCLUDE_DIRS}")
    include_directories(${GLFW_INCLUDE_DIRS})
    set (EXTRA_LIBS ${EXTRA_LIBS} ${GLFW_LIBRARIES})
endif (GLFW_FOUND)
endif (USE_GRAPHICS)

# create the cuda library
if (USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories("${PROJECT_SOURCE_DIR}/src/CudaFunctions")
    include_directories("${PROJECT_SOURCE_DIR}/src/CudaFunctions/cuHelpers")

    add_subdirectory("${PROJECT_SOURCE_DIR}/src/CudaFunctions")

    set (EXTRA_LIBS ${EXTRA_LIBS} CudaFunctions)
    include_directories("${CUDA_INCLUDE_DIRS}")

    # ignore warnings from 3rd party files
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/src/CudaFunctions/cuHelpers")
endif (USE_CUDA)

# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wcast-align -Wcast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self -Wmissing-declarations -Wmissing-include-dirs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Woverloaded-virtual -Wredundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default -Wundef -Werror -Wno-unused")

# make project into library that can be used by multiple executables (such as test classes)
# RES_FILES is ignored by cmake but they show up in the build tree
add_library (RendererLib ${SOURCE} ${HEADERS} ${RES_FILES})

add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_executable (Renderer "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries (Renderer RendererLib ${EXTRA_LIBS})







