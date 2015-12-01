
if (USE_CUDA)
    set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/example_unittest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mathFunctions_unittest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cudaIntersections_unittest.cpp"
        PARENT_SCOPE
    )
else (USE_CUDA)
    set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/example_unittest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mathFunctions_unittest.cpp"
        PARENT_SCOPE
    )
endif (USE_CUDA)
