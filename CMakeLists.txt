cmake_minimum_required (VERSION 2.6)
project (Renderer)

# for testing
include(CTest)

# The version number.
set (Renderer_VERSION_MAJOR 1)
set (Renderer_VERSION_MINOR 0)

if(MSVC)
	# Force to always compile with W4
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	# Update if necessary
        set(
            CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}
            -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy
            -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations
            -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual
            -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
            -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused"
        )
endif()

# should we use cuda functions?
option(USE_CUDA "Use CUDA kernels if available" ON)

# resource location
set(RESOURCES_PATH "${PROJECT_SOURCE_DIR}/res")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
	"${PROJECT_SOURCE_DIR}/src/RendererConfig.hpp.in"
        "${PROJECT_BINARY_DIR}/src/RendererConfig.hpp"
)

# add the binary tree to the search path for include files
# so that we will find RendererConfig.hpp
include_directories (
        "${PROJECT_BINARY_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src/application"
        "${PROJECT_SOURCE_DIR}/src/shared"
        "${PROJECT_SOURCE_DIR}/src/renderApp"
        "${PROJECT_SOURCE_DIR}/src/renderApp/graphics"
#        "${PROJECT_SOURCE_DIR}/src/glfw-3.1.2/include"
)

add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/application")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/shared")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/renderApp")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/renderApp/graphics")
#add_subdirectory("${PROJECT_SOURCE_DIR}/src/glfw-3.1.2")

# resources
add_subdirectory("${RESOURCES_PATH}/shaders")

# add glm functionality
find_package(GLM REQUIRED)
if (GLM_FOUND)
    message("-- Found GLM: ${GLM_INCLUDE_DIRS}")
    include_directories("${GLM_INCLUDE_DIRS}")
endif (GLM_FOUND)

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

# create the cuda library
if (USE_CUDA)
    include_directories ("${PROJECT_SOURCE_DIR}/src/CudaFunctions")
    add_subdirectory (src/CudaFunctions)
    set (EXTRA_LIBS ${EXTRA_LIBS} CudaFunctions)
endif (USE_CUDA)

include_directories("${CUDA_INCLUDE_DIRS}")

# add the executable
# RES_FILES is ignored by cmake but they show up in the build tree
add_executable (Renderer ${SOURCE} ${HEADERS} ${RES_FILES})
target_link_libraries (Renderer ${EXTRA_LIBS})

add_custom_command(TARGET Renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/res/shaders
        $<TARGET_FILE_DIR:Renderer>/shaders)

# add the install targets
install (TARGETS Renderer DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/src/RendererConfig.hpp"
	DESTINATION include)



# does the application run
add_test (RendererRuns Renderer -s)

#define a macro to simplify adding tests
macro (test_sum arg result)
	add_test (RendererSum${arg} Renderer -s ${arg})
	set_tests_properties (RendererSum${arg}
		PROPERTIES PASS_REGULAR_EXPRESSION ${result}
)
endmacro (test_sum)

# invalid input
test_sum (-15 "ERROR:")
test_sum (-1 "ERROR:")
test_sum (-2.34 "ERROR:")
test_sum (65536 "ERROR:") # higher than max valid input

# invalid input made valid
test_sum (2.34 "Using: 2")
test_sum (0.001 "Using: 0")
test_sum (90.210 "Using: 90")
test_sum (6.66666 "Using: 6")
test_sum (20394.857 "Using: 20394")

# valid sums
test_sum (0 "0 is 0") # min valid input
test_sum (1 "1 is 1")
test_sum (2 "2 is 3")
test_sum (3 "3 is 6")
test_sum (4 "4 is 10")
test_sum (5 "5 is 15")
test_sum (10 "10 is 55")

# these can take a while
test_sum (20485 "20485 is 209827855")
test_sum (420 "420 is 88410")
test_sum (12345 "12345 is 76205685")
test_sum (23456 "23456 is 275103696")
test_sum (34567 "34567 is 597456028")
test_sum (45678 "45678 is 1043262681")
test_sum (56789 "56789 is 1612523655")
test_sum (65534 "65534 is 2147385345")
test_sum (65535 "65535 is 2147450880") # max valid input








