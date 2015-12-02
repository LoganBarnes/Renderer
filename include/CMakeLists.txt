
if (USE_CUDA)
    set(HEADERS
        ${HEADERS}
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.hpp"
        # renderApp/graphics
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/graphicsHandler.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/camera.hpp"
        # shared
        "${CMAKE_CURRENT_SOURCE_DIR}/shared/renderObjects.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared/renderTypes.hpp"
        # cuda
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/intersections.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CudaFunctions.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/tester.cuh"
        PARENT_SCOPE
    )
else (USE_CUDA)
    set(HEADERS
        ${HEADERS}
        # renderApp
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.hpp"
        # renderApp/graphics
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/graphicsHandler.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/camera.hpp"
        # shared
        "${CMAKE_CURRENT_SOURCE_DIR}/shared/renderObjects.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared/renderTypes.hpp"
    )
endif (USE_CUDA)
