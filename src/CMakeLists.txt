# A list of all the files needed to
# compile the RenderLib library

if (USE_CUDA)
    set(SOURCE
        ${SOURCE}
        # glfwApp
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/graphicsHandler.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/camera.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callbackSingleton.cpp"
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderCallback.cpp"
        PARENT_SCOPE
    )
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
else (USE_CUDA)
    set(SOURCE
        ${SOURCE}
        # glfwApp
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/graphicsHandler.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/camera.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glfwApp/callbackSingleton.cpp"
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderCallback.cpp"
        #renderApp/cpu
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/renderRandom.cpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/intersections.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/pathTracer.cpp"
        #testing
#        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/cpu/tester.cpp"
        PARENT_SCOPE
    )
endif (USE_CUDA)

