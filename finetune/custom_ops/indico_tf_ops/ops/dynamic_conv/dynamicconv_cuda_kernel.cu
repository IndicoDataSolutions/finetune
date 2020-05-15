/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamicconv.h"
#include "cuda_utils.cu"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_device_functions.h"

template<typename scalar_t>
__global__
void dynamicconv_forward_kernel(const scalar_t* input, // batch, chanels, sequence
                                const scalar_t* weight, // ?, numHeads, filterSize
				scalar_t* temp_input_full, // (n_features * batch_size) * (block_dim + fs)
				scalar_t* filter_full, // (n_features * batch_size) * n_threads * fs
				int filterSize,
				int padding_l,
                                int minibatch,
                                int sequenceLength,
                                int numFeatures,
                                int numFiltersInBlock,
                                int numHeads,
                                scalar_t* output) {
//  assert(blockDim.x == SB);

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int head = featureIdx / numFiltersInBlock;
  const int tempInputOffset = batchIdx * numFeatures * (blockDim.x + filterSize)
                            + featureIdx * (blockDim.x + filterSize);
		       
  const int filterOffset = batchIdx * numFeatures * blockDim.x * filterSize
       	               + featureIdx * blockDim.x * filterSize
		       + tid * filterSize;
		       
  const int IOOffset = batchIdx * numFeatures * sequenceLength
                       + featureIdx * sequenceLength;
		       
  const scalar_t* inputFeature = &input[IOOffset];
  scalar_t* outputFeature = &output[IOOffset];

  scalar_t* tempInput = &temp_input_full[tempInputOffset];
  scalar_t* filter = &filter_full[filterOffset];
  
  zeroSharedMem<scalar_t>(tempInput, padding_l, filterSize);

  const int numIterations = divUp<int, int>(sequenceLength, blockDim.x);

  for (int i = 0; i < numIterations; ++i) {
    __syncthreads();
    const int inputOffset = i * blockDim.x;
    load_input_to_shared<scalar_t>(inputFeature, inputOffset,
                        	   sequenceLength, i,
                                   numIterations, false, tempInput,
				   filterSize, padding_l);
    __syncthreads();
    if (inputOffset + tid < sequenceLength) {

      #pragma unroll
      for (int k = 0; k < filterSize; ++k) {
        const int filterOffset = batchIdx * numHeads * filterSize * sequenceLength
                                 + head * filterSize * sequenceLength
                                 + k * sequenceLength
                                 + i * blockDim.x + tid;
        filter[k] = weight[filterOffset];
      }

      scalar_t out = scalar_t(0.0);
      #pragma unroll
      for (int k = 0; k < filterSize; ++k) {
        out += filter[k] * tempInput[tid + k];
      }

      outputFeature[inputOffset + tid] = out;

    }
  }
}

template<typename scalar_t>
__global__
void dynamicconv_backward_kernel(
    const scalar_t* gradOutput, // B * C * T
    const scalar_t* input, // B * C * T
    const scalar_t* weight,
    scalar_t* tempGradOutputFull, // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
    scalar_t* tempInputFull, // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
    scalar_t* tempGradSumFull, // minibatch, numHeads, numChunks, num_threads, filterSize
    scalar_t* bfilterFull, // minibatch, numHeads, numChunks, num_threads, filterSize
    int filterSize,
    int padding_l,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    scalar_t* gradWeight,
    scalar_t* gradInput) { // B * H * k * T

//  assert(blockDim.x == SB);

  // each block operates on a single batch and filter head
  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int headIdx = blockIdx.y;
  const int chunkIdx = blockIdx.z;

  const int numChunks = divUp<int, int>(sequenceLength, blockDim.x);
  const int inputOffset = chunkIdx * blockDim.x;

  // initialize shared memory for output gradient and input
  const int tempInputOffset = batchIdx * numHeads * numChunks * (blockDim.x + filterSize)
                            + headIdx * numChunks * (blockDim.x + filterSize)
			    + chunkIdx * (blockDim.x + filterSize);
			    
  scalar_t* tempGradOutput = &tempGradOutputFull[tempInputOffset];
  scalar_t* tempInput = &tempInputFull[tempInputOffset];
  const int padding = filterSize - padding_l - 1;

  zeroSharedMem(tempGradOutput, padding, filterSize);
  zeroSharedMem(tempInput, padding_l, filterSize);

  // initialize local filter and weight gradient sum arrays
  const int tempGradSumOffset = batchIdx * numHeads * numChunks * blockDim.x * filterSize
                            + headIdx * numChunks * blockDim.x * filterSize
			    + chunkIdx * blockDim.x * filterSize
			    + tid * filterSize;
				  
  scalar_t* tempGradSum = &tempGradSumFull[tempGradSumOffset];
  scalar_t* bfilter = &bfilterFull[tempGradSumOffset];

  for (int k = 0; k < filterSize; ++k) {
    tempGradSum[k] = scalar_t(0.0);

    int idxOffset = inputOffset + tid + k - padding;
    if (idxOffset >= 0 && idxOffset < sequenceLength) {
      int bfilterOffset = batchIdx * numHeads * filterSize * sequenceLength
                          + headIdx * filterSize * sequenceLength
                          + (filterSize - k  - 1) * sequenceLength
                          + idxOffset;
      bfilter[k] = weight[bfilterOffset];
    } else {
      bfilter[k] = scalar_t(0.0);
    }
  }


  // iterate over filter block
  for (int featureIdx = 0; featureIdx < numFiltersInBlock; ++featureIdx) {
    __syncthreads();

    // load input and output gradient for this channel and chunk
    const int IOOffset = batchIdx * numFeatures * sequenceLength
                         + (headIdx * numFiltersInBlock + featureIdx) * sequenceLength;
    const scalar_t* inputFeature = &input[IOOffset];
    const scalar_t* gradOutputFeature = &gradOutput[IOOffset];
    scalar_t* gradInputFeature = &gradInput[IOOffset];

    load_input_to_shared(gradOutputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempGradOutput,
					    filterSize, padding);
    load_input_to_shared(inputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempInput,
					    filterSize, padding_l);
    __syncthreads();
 
    // sum input and weight gradients
    scalar_t out = scalar_t(0.0);
    #pragma unroll
    for (int k = 0; k < filterSize; ++k) {
      tempGradSum[k] += tempInput[tid + k] * tempGradOutput[tid + padding];
      out += bfilter[k] * tempGradOutput[tid + k];
    }
    
    if (inputOffset + tid < sequenceLength) {
      gradInputFeature[inputOffset + tid] = out;
    }
  }

  const int gradOffset = batchIdx * numHeads * filterSize * sequenceLength
               + headIdx * filterSize * sequenceLength;
  scalar_t *gradWeightFeature = &gradWeight[gradOffset];

  // write weight gradient
  if (inputOffset + tid < sequenceLength) {
    for (int k = 0; k < filterSize; ++k) {
      const int outputOffset = k * sequenceLength + inputOffset + tid;
      gradWeightFeature[outputOffset] = tempGradSum[k];
    }
  }
}


template <typename scalar_t>
void DynamicConvForwardLauncher(const scalar_t* input, // batch, chanels, sequence
				const scalar_t* weight, // ?, numHeads, filterSize
				scalar_t* temp_input_full, // (n_features * batch_size) * (block_dim + fs)
				scalar_t* filter_full, // (n_features * batch_size) * n_threads * fs
				int filterSize,
				int padding_l,
				int minibatch,
				int sequenceLength,
				int numFeatures,
				int numFiltersInBlock,
				int numHeads,
				scalar_t* output,
				const dim3 blocks,
				int b_size)
{
  dynamicconv_forward_kernel<scalar_t><<<blocks, b_size>>>(
  					      input, // batch, chanels, sequence
                                  	      weight, // ?, numHeads, filterSize
				              temp_input_full, // (n_features * batch_size) * (block_dim + fs)
					      filter_full, // (n_features * batch_size) * n_threads * fs
					      filterSize,
					      padding_l,
					      minibatch,																				                                        sequenceLength,																								          numFeatures,
					      numFiltersInBlock,
					      numHeads,
					      output);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}


template <typename scalar_t>
void DynamicConvBackwardLauncher(const scalar_t* gradOutput, // B * C * T
				const scalar_t* input, // B * C * T
				const scalar_t* weight,
				scalar_t* tempGradOutputFull, // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
				scalar_t* tempInputFull, // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
				scalar_t* tempGradSumFull, // minibatch, numHeads, numChunks, num_threads, filterSize
				scalar_t* bfilterFull, // minibatch, numHeads, numChunks, num_threads, filterSize
				int filterSize,
				int padding_l,
				int minibatch,
				int sequenceLength,
				int numFeatures,
				int numFiltersInBlock,
				int numHeads,
				scalar_t* gradWeight,
				scalar_t* gradInput,
				const dim3 blocks,
				int b_size)
{
  dynamicconv_backward_kernel<scalar_t><<<blocks, b_size>>>(gradOutput, input, weight,
				       tempGradOutputFull, tempInputFull, tempGradSumFull, bfilterFull,
				       filterSize, padding_l, minibatch, sequenceLength,
				       numFeatures, numFiltersInBlock, numHeads,
				       gradWeight, gradInput);
    
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
				
#define DEFINE_GPU_KERNELS(scalar_t)\
  template void DynamicConvForwardLauncher<scalar_t>(const scalar_t* ,const scalar_t*, scalar_t*, scalar_t*, int, int, int, int, int, int, int, scalar_t*, const dim3, int);\
  template void DynamicConvBackwardLauncher<scalar_t>(const scalar_t* ,const scalar_t* ,const scalar_t* ,scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, int, int, int, int, int, int, scalar_t*, scalar_t*, const dim3, int);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
