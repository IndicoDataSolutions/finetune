cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)


execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_INCLUDE_LIBS)

execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)


execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TF_CFLAGS)

execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TF_LFLAGS)


message("Tensorflow_INCLUDE_DIRS " ${Tensorflow_INCLUDE_DIRS})
message("Tensorflow_INCLUDE_LIBS " ${Tensorflow_INCLUDE_LIBS})

message("TF_CFLAGS " ${TF_CFLAGS})
message("TF_LFLAGS " ${TF_LFLAGS})

find_package(CUDA)

# C++14 required for tensorflow
#set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${TF_CFLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS " -DNDEBUG -Wl,--no-as-needed ${TF_LFLAGS} ${CMAKE_CXX_FLAGS}")

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

find_package( CUDA REQUIRED )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -v" )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,\"code=sm_75,compute_75\" " )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_70,\"code=sm_70,compute_70\" " )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_62,\"code=sm_62,compute_62\" " )
# Lowest supported by cuda 12 - get as broad a support as possible.
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,\"code=sm_60,compute_60\" " )
set( CUDA_NVCC_FLAGS "${TF_CFLAGS} ${CUDA_NVCC_FLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC")
message("CUDA_NVCC_FLAGS " ${CUDA_NVCC_FLAGS})

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS}/)

FILE(COPY
  ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h
  ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
  ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_fp16.h
  ${CUDA_TOOLKIT_TARGET_DIR}/include/device_functions.h
  ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
  DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
)
include_directories(${tensorflow_source_dir})

#create library
cuda_add_library(
    indico_kernels SHARED
    ops/ra/cuda_op_kernel.cu
    ops/ra/ra.cc
    ops/dynamic_conv/dynamicconv.cc
    ops/dynamic_conv/dynamicconv_cuda_kernel.cu
    ops/memory/memory_ops.cc
    ops/memory/memory.cc
)
