# CMakeLists.txt to build CUDA program from cuda.cu

set(CUDA_SOURCE
    ${CUDA_SOURCE}
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/random.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/random_kernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/intersections.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/pathTracer.cu"
#testing
    "${CMAKE_CURRENT_SOURCE_DIR}/tester.cu"
)


set(SOURCE
    ${SOURCE}
    ${CUDA_SOURCE}
    PARENT_SCOPE
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fno-strict-aliasing -use_fast_math")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")

cuda_add_library(CudaLib ${CUDA_SOURCE})
target_link_libraries(CudaLib ${CUDA_curand_LIBRARY})

set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)








# include(FindCUDA)

# add the binary tree directory to the search path 
# for include files
# include_directories( ${CMAKE_CURRENT_BINARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/include)


    # NVCCFLAGS = -Xlinker -rpath,$$CUDA_LIB
# -m64 -O3 -arch=sm_30 -c $$NVCCFLAGS
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30--std=c++11
	# --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
	# -Xlinker -rpath, ${CUDA_TOOLKIT_ROOT_DIR}/lib)
# list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# add the main library
# cuda_add_library(
# 	CudaLib
# 	cuda.cu)

# target_link_libraries(CudaLib
	# -lcudart
	# )#${CMAKE_CURRENT_BINARY_DIR}/libcudart.so)

# install (TARGETS CudaLib DESTINATION bin)
# install (FILES CudaLib.h DESTINATION include)


    # # Cuda sources
    # CUDA_SOURCES += cuda/wrappers.cu

    # # show files in working tree
    # OTHER_FILES +=  cuda/wrappers.cu \
    #                 cuda/wrappers.cuh \
    #                 cuda/helper_cuda.h

    # # Path to cuda install
    # CUDA_LIB = $$CUDA_DIR/lib

    # # Pather to header and lib files
    # INCLUDEPATH += $$CUDA_DIR/include \
    #                cuda # my cuda files
    # QMAKE_LIBDIR += $$CUDA_LIB

    # # prevents warnings from code we didn't write
    # QMAKE_CXXFLAGS += -isystem $$CUDA_DIR/include

    # LIBS += -lcudart # add other cuda libs here (-lcublas -lcurand, etc.)

    # # SPECIFY THE R PATH FOR NVCC!!!!! (your problem...previously my problem)
    # QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_LIB
    # NVCCFLAGS = -Xlinker -rpath,$$CUDA_LIB

    # # libs used in the code
    # CUDA_LIBS = $$LIBS
    # CUDA_LIBS -= $$NON_CUDA_LIBS # remove libs nvcc won't recognize

    # # GPU architecture (might be a way to detect this somehow instead of hardcoding)
    # CUDA_ARCH     = sm_30 # <- based on specs from your code. This was tested with sm_30

    # # Some default NVCC flags
    # NVCCFLAGS     += --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v --std=c++11

    # # Prepare the extra compiler configuration (taken from the nvidia forum)
    # CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

    # cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
    #                 $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
    #                 $$SED_STUFF
    # # nvcc error printout format ever so slightly different from gcc
    # # http://forums.nvidia.com/index.php?showtopic=171651

    # cuda.dependency_type = TYPE_C
    # cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

    # cuda.input = CUDA_SOURCES
    # cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

    # # Tell Qt that we want add more stuff to the Makefile
    # QMAKE_EXTRA_COMPILERS += cuda
