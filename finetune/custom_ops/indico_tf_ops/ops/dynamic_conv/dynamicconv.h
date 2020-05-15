/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef DYNAMIC_CONV_H
#define DYNAMIC_CONV_H

#define EIGEN_USE_GPU
#define GOOGLE_CUDA 1

#include "tensorflow/core/framework/register_types.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include<bits/stdc++.h>
#define MAX_THREADS_PER_BLOCK 256
#define WARP_SIZE 32

using namespace tensorflow;

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
				 int b_size);


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
				int b_size);

int nextPowerOf2(int n);

#endif /* DYNAMIC_CONV_H */
