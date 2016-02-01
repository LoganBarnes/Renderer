cmake_minimum_required (VERSION 3.0.0)
project (Renderer)

# The version number.
set (Renderer_VERSION_MAJOR 1)
set (Renderer_VERSION_MINOR 0)

# should we use cuda functions?
option(USE_CUDA "Use CUDA kernels if available" OFF)

# should we show the output graphically?
option(USE_GRAPHICS "Use CUDA kernels if available" ON) # haven't implemented 'OFF' version yet

# resource location
set(RESOURCES_PATH "${PROJECT_SOURCE_DIR}/res")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
    "${PROJECT_SOURCE_DIR}/src/renderApp/renderer-config.hpp.in"
    "${PROJECT_BINARY_DIR}/include/renderApp/renderer-config.hpp"
)

# add the binary tree to the search path for include files
# so that we will find renderer-config.hpp
include_directories (
    "${PROJECT_BINARY_DIR}/include/renderApp"
    "${PROJECT_SOURCE_DIR}/include/renderApp"
    "${PROJECT_SOURCE_DIR}/include/glfwApp"
)

# set source files
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

## add glm functionality
set(GLM_DIR "${PROJECT_SOURCE_DIR}/../thirdparty/glm")
if (EXISTS ${GLM_DIR})
    message("-FOUND: GLM")
    add_subdirectory("${GLM_DIR}" "${PROJECT_SOURCE_DIR}/../thirdparty/bin/glm")
    include_directories("${GLM_DIR}/glm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GLM_DIR}/glm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GLM_DIR}/glm/detail")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GLM_DIR}/glm/gtx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GLM_DIR}/glm/gtc")
else()
    message("Couldn't find the GLM library: Make sure the thirdparty directory is located in the same folder as the project directory")
endif()

if (USE_GRAPHICS)
    # add opengl functionality
    find_package(OpenGL REQUIRED)
    if (OPENGL_FOUND)
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
    if (EXISTS "${PROJECT_SOURCE_DIR}/../thirdparty/glfw")
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/../thirdparty/glfw" # glfw src location
            "${PROJECT_SOURCE_DIR}/../thirdparty/bin/glfw" # glfw binary location
            )
        include_directories("${PROJECT_SOURCE_DIR}/../thirdpart/glfw/include")
        set (EXTRA_LIBS ${EXTRA_LIBS} glfw ${GLFW_LIBRARIES})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/../thirdpart/glfw/include")
    else ()
        message("Couldn't find the GLFW library: Make sure the thirdparty directory is located in the same folder as the project directory")
    endif ()


else (USE_GRAPHICS)
    # use ncurses terminal functionality
    set(CURSES_NEED_NCURSES TRUE)
    find_package(Curses REQUIRED)
    if (CURSES_FOUND)
        include_directories(${CURSES_INCLUDE_DIR})
        set (EXTRA_LIBS ${EXTRA_LIBS} ${CURSES_LIBRARY})
    endif (CURSES_FOUND)

endif (USE_GRAPHICS)

if (USE_CUDA)
    # create the cuda library
    find_package(CUDA REQUIRED)
    include_directories("${PROJECT_SOURCE_DIR}/include/helperCuda")
    include_directories("${PROJECT_SOURCE_DIR}/include/renderApp/cuda")

    # set nvcc options
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_30,code=sm_30")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fno-strict-aliasing -use_fast_math")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")

    # build CUDA library
    cuda_add_library(CudaLib ${CUDA_SOURCE})
    target_link_libraries(CudaLib ${CUDA_curand_LIBRARY})

    set (EXTRA_LIBS ${EXTRA_LIBS} CudaLib)
    include_directories("${CUDA_INCLUDE_DIRS}")

    # ignore warnings from 3rd party files
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/include/helperCuda")

else (USE_CUDA)
    # use pthread library
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads REQUIRED)

    include_directories("${PROJECT_SOURCE_DIR}/include/renderApp/cpu")
    set (EXTRA_LIBS ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})

endif (USE_CUDA)

# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wcast-align -Wcast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self -Wmissing-declarations -Wmissing-include-dirs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Woverloaded-virtual -Wredundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default -Wundef -Werror -Wno-unused")

# make project into library that can be used by multiple executables (such as test classes)

# USED WITH QT CREATOR
add_subdirectory("${PROJECT_SOURCE_DIR}/include") # QT
add_subdirectory("${RESOURCES_PATH}") # QT
add_library (RendererLib ${SOURCE} ${HEADERS} ${RES_FILES} ${CUDA_SOURCE}) # QT
# ELSE (WHEN NOT USING QT CREATOR)
#add_library (RendererLib ${SOURCE}) # NON-QT

add_executable (Renderer "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries (Renderer RendererLib ${EXTRA_LIBS})







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
#         # Make sure lib is linked to each app
#         target_link_libraries( ${testname} RendererLib ${EXTRA_LIBS} ${GTEST_BOTH_LIBRARIES} ${GMOCK_BOTH_LIBRARIES})
#     endforeach( testsourcefile ${TEST_SOURCES} )
# endif (GTEST_FOUND)




