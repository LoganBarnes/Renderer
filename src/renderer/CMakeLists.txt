
set(SOURCE
	${SOURCE}
        "${CMAKE_CURRENT_SOURCE_DIR}/glHandler.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathTracer.cpp"
	PARENT_SCOPE
)

set(HEADERS
	${HEADERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/glHandler.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/renderApp.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathTracer.hpp"
	PARENT_SCOPE
)
