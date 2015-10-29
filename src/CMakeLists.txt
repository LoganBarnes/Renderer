
set(SOURCE
	${SOURCE}
	"${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GLFWApplication.cpp"
	PARENT_SCOPE
)

set(HEADER
        ${HEADER}
        "${CMAKE_CURRENT_SOURCE_DIR}/GLFWApplication.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Application.hpp"
        PARENT_SCOPE
)

