/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


template <typename U, typename V>	
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {	
  return (a + b - 1) / b;	
}


template<typename scalar_t>
__inline__ __device__
void zeroSharedMem(scalar_t* data, int padding_l, int filter_size) {
  /*
    Given an array of length filter_size + blockDim.x, zero out the first padding_l and last
    (filter_size - padding_l) values in the array
  */
  int tid = threadIdx.x;

  if (filter_size < blockDim.x) {
    // zero all if we have enough threads in a block to do all of them
    if (tid < padding_l || tid > blockDim.x - filter_size + padding_l - 1) {
      data[tid] = scalar_t(0.0);
    }
  } else {
    // otherwise zero out one block at a time
    const int numIterations = divUp<int, int>(filter_size, blockDim.x);
    for (int i = 0; i < numIterations; i++) {
      int offset = i * blockDim.x;
      if (tid + offset < padding_l) {
        data[tid + offset] = scalar_t(0.0);
      } else if (tid + offset < filter_size) {
        data[blockDim.x + tid + offset] = scalar_t(0.0);
      }
    }
  }
}


void checkCudaStatus(cudaError_t status, int lineNumber = -1) {

  if (status != cudaSuccess) {
    std::cout << cudaGetErrorString(status)
              << " at line " << lineNumber << std::endl;
    std::cout << "Exiting" << std::endl;
    exit(1);
  }
}

template<typename scalar_t>
__device__
void load_input_to_shared(const scalar_t* input, // global memory
                          int inputOffset, int sequenceLength,
                          int iteration, int numIterations,
                          bool no_prev, scalar_t* output, /* shared memory */
			  int filterSize, int padding_l) {
  /*
    Load a block size of input into shared memory with
    right and left overhang of total size FS. If previously
    loaded memory, overlap will be shifted over to reduce
    global memory access

    input - pointer to start of channel sequence
    inputOffset - how far in the sequence to start loading
    sequenceLength - total length of sequence
    iteration - which block of sequence we are loading
    numIterations - total number of blocks to load
    no_prev - whether to load the whole block if the previous block
              wasn't loaded
    output - shared memory to write input to
  */

  const int tid = threadIdx.x;

  // Load the left "overhang" of input
  if (iteration > 0) {
    if (padding_l < blockDim.x) {

      // load all at once
      if (tid < padding_l) {
        output[tid] = (no_prev) ? input[inputOffset - padding_l + tid] : output[tid + blockDim.x];
      }
    } else {

      // load in chunks of size SB
      int numIterations = divUp<int, int>(padding_l, blockDim.x);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * blockDim.x;
        if ((tid + offset) < padding_l) {
          output[tid + offset] = (no_prev) ? input[inputOffset - padding_l + tid + offset] : output[tid + offset + blockDim.x];
        }
      }
    }
  }

  // Load the right "overhang" of input
  if (iteration < (numIterations - 1)) {
    const int elementsLeft = sequenceLength - (iteration+1) * blockDim.x;

    if ((filterSize - padding_l) < blockDim.x) {

      // load all at once
      if (tid < (filterSize - padding_l)) {
          output[padding_l + blockDim.x + tid] = (tid < elementsLeft) ? input[inputOffset + blockDim.x + tid] : scalar_t(0.0);
      }
    } else {
      // load in chunks of size blockDim.x
      int numIterations = divUp<int, int>(filterSize - padding_l, blockDim.x);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * blockDim.x;
        if ((tid + offset) < (filterSize - padding_l)) {
          output[padding_l + blockDim.x + tid + offset] = ((tid + offset) < elementsLeft) ? input[inputOffset + blockDim.x + tid + offset] : scalar_t(0.0);
        }
      }
    }
  }

  // We should also clear out the right "overhang"
  if (iteration == (numIterations - 1)) {
    if ((filterSize - padding_l) < blockDim.x) {

      // clear out all at once
      if (tid < (filterSize - padding_l)) {
          output[padding_l + blockDim.x + tid] = scalar_t(0.0);
      }
    } else {

      // clear in chunks of size blockDim.x
      int numIterations = divUp<int, int>(filterSize - padding_l, blockDim.x);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * blockDim.x;
        if ((tid + offset) < (filterSize - padding_l)) {
          output[padding_l + blockDim.x + tid + offset] = scalar_t(0.0);
        }
      }
    }
  }
  output[tid + padding_l] = ((inputOffset + tid) < sequenceLength) ? input[inputOffset + tid] : scalar_t(0.0);
}
