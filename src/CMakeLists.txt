# A list of all the files needed to
# compile the RenderLib library
set(SOURCE
    ${SOURCE}
    # renderApp
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/renderApp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/pathTracer.cpp"
    # renderApp/graphics
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/graphicsHandler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/renderApp/graphics/camera.cpp"
    PARENT_SCOPE
)

