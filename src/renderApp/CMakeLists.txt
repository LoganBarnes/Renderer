
set(SOURCE
        ${SOURCE}
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathTracer.cpp"
	PARENT_SCOPE
)

set(HEADERS
        ${HEADERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathTracer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderObjects.hpp"
	PARENT_SCOPE
)
