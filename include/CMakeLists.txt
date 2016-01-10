# A list of all the header files so they show up
# in the project tree when using Qt Creator
if (USE_CUDA)
    set(HEADERS
        ${HEADERS}
        # glfwApp
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/graphicsHandler.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/camera.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callbackSingleton.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callback.hpp"
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderCallback.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderObjects.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderTypes.hpp"
        # renderApp/cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/cuda_render.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/cuda_tester.cuh"
        # helperCuda
        "${CMAKE_CURRENT_SOURCE_DIR}/helperCuda/cuda_wrappers.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/helperCuda/cuda_random.cuh"
        PARENT_SCOPE
    )
else (USE_CUDA)
    set(HEADERS
        ${HEADERS}
        # glfwApp
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/graphicsHandler.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/camera.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callbackSingleton.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callback.hpp"
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderCallback.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderObjects.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderTypes.hpp"
        # renderApp/cpu
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/cpu_render.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/cpu_tester.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/cpu_random.hpp"
        PARENT_SCOPE
    )
endif (USE_CUDA)
