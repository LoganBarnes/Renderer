# A list of all the files needed to
# compile the RenderLib library
set(SOURCE
    ${SOURCE}
    # glfwApp
    "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/graphicsHandler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/camera.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/input.cpp"
    # renderApp
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.cpp"
    PARENT_SCOPE
)

if (USE_CUDA)
    set(CUDA_SOURCE
        ${CUDA_SOURCE}
        # helperCuda
        "${CMAKE_CURRENT_SOURCE_DIR}/helperCuda/wrappers.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/helperCuda/random.cu"
        # renderApp/cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/renderRandom.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/intersections.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/pathTracer.cu"
        #testing
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cuda/tester.cu"
        PARENT_SCOPE
    )
endif (USE_CUDA)

