cmake_minimum_required ( VERSION 3.2.0 )
project ( Renderer )

# namespace used for project
set ( PROJECT_NAMESPACE rndr )

# The version number.
set ( PROJECT_VERSION_MAJOR 1 )
set ( PROJECT_VERSION_MINOR 0 )
set ( PROJECT_VERSION_PATCH 0 )

# required before DefaultProjectLibrary.cmake
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src )

# other paths
set( SHARED_PATH ${PROJECT_SOURCE_DIR}/shared )
set( RES_PATH    ${PROJECT_SOURCE_DIR}/res    )
set( SHADER_PATH ${SHARED_PATH}/shaders       )

set( PROJECT_CONFIG_FILE ${SHARED_PATH}/src/common/ProjectConfig.hpp.in )


set( USE_GLFW ON CACHE BOOL "" FORCE )
set( USE_GLM  ON CACHE BOOL "" FORCE )

# set variables from shared simulation project
add_subdirectory( ${SHARED_PATH} )

# system header dirs
set(
    PROJECT_SYSTEM_INCLUDE_DIRS
    ${SHARED_SYSTEM_INCLUDE_DIRS}
    )

# libraries to link against
set(
    PROJECT_LINK_LIBS
    ${SHARED_LINK_LIBS}
    )

# must be built before project lib
set(
    PROJECT_DEP_TARGETS
    ${SHARED_DEP_TARGETS}
    )



set(
    PROJECT_CUDA_SOURCE

    ${SHARED_PATH}/src/cuda/CudaWrappers.cuh
    ${SHARED_PATH}/src/cuda/CudaWrappers.cu
    ${SHARED_PATH}/src/cuda/CudaRandom.cuh
    ${SHARED_PATH}/src/cuda/CudaRandom.cu

    ${SRC_DIR}/path/cuda/CudaPathTracer.cuh
    ${SRC_DIR}/path/cuda/CudaPathTracer.cu
    ${SRC_DIR}/path/cuda/CudaRendererRandom.cu
    ${SRC_DIR}/path/cuda/CudaIntersections.cu
    )

set(
    PROJECT_CUDA_INCLUDE_DIRS

    ${SHARED_INCLUDE_DIRS}

    ${SHARED_PATH}/src/cuda
    ${SRC_DIR}/path/cuda
    ${SRC_DIR}/common
    )

set(
    PROJECT_CUDA_SYSTEM_INCLUDE_DIRS

    ${SHARED_SYSTEM_INCLUDE_DIRS}

    ${SHARED_PATH}/src/cuda/helpers
    )


set(
    CUDA_COMPUTE_FLAGS
    "-gencode arch=compute_30,code=sm_30"
    " -gencode arch=compute_35,code=sm_35"
    )

set( USE_CUDA ON )


# header dirs
set(
    PROJECT_INCLUDE_DIRS

    ${SHARED_INCLUDE_DIRS}

    ${SRC_DIR}/common
    ${SRC_DIR}/io
    ${SRC_DIR}/kdTree
    ${SRC_DIR}/path
    ${SRC_DIR}/path/cuda
    ${SRC_DIR}/shapes
    ${SRC_DIR}/world
    )

# cpp files
set(
    PROJECT_SOURCE

    # ignored by compilers but added to project files
    ${SHARED_UNCRUSTIFY_FILE}
    ${SHADER_PATH}/screenSpace/shader.vert
    ${SHADER_PATH}/blending/shader.frag

    ${SHARED_SOURCE}

    ${SRC_DIR}/common/RenderTypes.hpp
    ${SRC_DIR}/common/RenderObjects.hpp

    ${SRC_DIR}/io/RendererIOHandler.hpp
    ${SRC_DIR}/io/RendererIOHandler.cpp
    ${SRC_DIR}/io/RendererCallback.hpp
    ${SRC_DIR}/io/RendererCallback.cpp

    ${SRC_DIR}/kdTree/KDTree.hpp
#    ${SRC_DIR}/kdTree/KDTree.cpp
    ${SRC_DIR}/kdTree/KDNode.hpp
#    ${SRC_DIR}/kdTree/KDNode.cpp
    ${SRC_DIR}/kdTree/KDElement.hpp

    ${SRC_DIR}/path/PathTracer.hpp
    ${SRC_DIR}/path/PathTracer.cpp

    ${SRC_DIR}/shapes/MinimalShapes.hpp
    ${SRC_DIR}/shapes/AABB.hpp
    ${SRC_DIR}/shapes/AABB.cpp

    ${SRC_DIR}/world/SceneWorld.hpp
    ${SRC_DIR}/world/SceneWorld.cpp

    )


# file with main function
set( PROJECT_MAIN ${SRC_DIR}/exec/Renderer.cpp )

include( ${SHARED_PATH}/cmake/DefaultProjectLibrary.cmake )















# # testing
# find_package(GTEST)
# if (GTEST_FOUND)
#     include_directories(${GTEST_INCLUDE_DIRS})

#     # GMock
#     include_directories("/usr/local/include/gmock")
#     set(GMOCK_BOTH_LIBRARIES "/usr/local/lib/libgmock.a" "/usr/local/lib/libgmock_main.a")

#     add_subdirectory("${PROJECT_SOURCE_DIR}/src/testing")
#     foreach( testsourcefile ${TEST_SOURCES} )

#         # I used a simple string replace, to cut off .cpp and path.
#         string( REPLACE ".cpp" "" testname ${testsourcefile} )
#         string( REPLACE "${PROJECT_SOURCE_DIR}/src/testing/" "" testname ${testname} )
#         add_executable( ${testname} ${testsourcefile} )
#         add_dependencies(${testname} ${DEPENDENCIES})

#         # Make sure lib is linked to each app
#         target_link_libraries( ${testname} RendererLib ${EXTRA_LIBS} ${GTEST_BOTH_LIBRARIES} ${GMOCK_BOTH_LIBRARIES})

#    endforeach( testsourcefile ${TEST_SOURCES} )

#endif (GTEST_FOUND)

