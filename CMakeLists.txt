
#    # create the cuda library
#    find_package(CUDA REQUIRED)
#    include_directories("${PROJECT_SOURCE_DIR}/include/helperCuda")
#    include_directories("${PROJECT_SOURCE_DIR}/include/renderApp/cuda")

#    # set nvcc options
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_30,code=sm_30")
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fno-strict-aliasing -use_fast_math")
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")

#    # build CUDA library
#    cuda_add_library(CudaLib ${CUDA_SOURCE})
#    target_link_libraries(CudaLib ${CUDA_curand_LIBRARY})

#    set (EXTRA_LIBS ${EXTRA_LIBS} CudaLib)
#    set (DEPENDENCIES ${DEPENDENCIES} CudaLib)

#    include_directories("${CUDA_INCLUDE_DIRS}")

#    # ignore warnings from 3rd party files
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/include/helperCuda")



## compile flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wcast-align -Wcast-qual")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self -Wmissing-declarations -Wmissing-include-dirs")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Woverloaded-virtual -Wredundant-decls")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default -Wundef -Werror -Wno-unused")

## make project into library that can be used by multiple executables (such as test classes)

## USED WITH QT CREATOR
#add_subdirectory("${PROJECT_SOURCE_DIR}/include") # QT
#add_subdirectory("${RESOURCES_PATH}") # QT
#add_library (RendererLib ${SOURCE} ${HEADERS} ${RES_FILES} ${CUDA_SOURCE}) # QT
## ELSE (WHEN NOT USING QT CREATOR)
##add_library (RendererLib ${SOURCE}) # NON-QT
#add_dependencies(RendererLib ${DEPENDENCIES})
#set (DEPENDENCIES ${DEPENDENCIES} RendererLib)

#add_executable (Renderer "${PROJECT_SOURCE_DIR}/src/main.cpp")
#target_link_libraries (Renderer RendererLib ${EXTRA_LIBS})
#add_dependencies(Renderer ${DEPENDENCIES})









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


# header dirs
set(
    PROJECT_INCLUDE_DIRS

    ${SHARED_INCLUDE_DIRS}
    )

# cpp files
set(
    PROJECT_SOURCE

    ${SHARED_SOURCE}
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

