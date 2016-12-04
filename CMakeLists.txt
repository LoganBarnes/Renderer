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
#set( USE_GLM  ON CACHE BOOL "" FORCE )

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

    ${SHARED_PATH}/src/cuda/CudaWrappers.cu
    ${SHARED_PATH}/src/cuda/CudaRandom.cu
    )

set(
    PROJECT_CUDA_INCLUDE_DIRS

    ${SHARED_PATH}/src/cuda
    )


set(
    CUDA_COMPUTE_FLAGS
    "-gencode arch=compute_30,code=sm_30"
    " -gencode arch=compute_35,code=sm_35"
    )


# header dirs
set(
    PROJECT_INCLUDE_DIRS

    ${SHARED_INCLUDE_DIRS}

    ${SRC_DIR}/io
    )

# cpp files
set(
    PROJECT_SOURCE

    ${SHARED_SOURCE}

    ${SRC_DIR}/io/RendererIOHandler.cpp
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

