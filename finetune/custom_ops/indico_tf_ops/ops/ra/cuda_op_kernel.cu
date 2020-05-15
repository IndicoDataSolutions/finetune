#define EIGEN_USE_GPU
#define GOOGLE_CUDA 1

#include "ra.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_device_functions.h"

#define THREADS_PER_BLOCK 512

using namespace tensorflow;

template <typename T>
__global__ void DenseKernel(
        const T* inputs,
        T* output,
        int* indexes,
        const int batch_samples,
        const int sequence_len,
        const int output_depth,
        const int kernel_size,
        const int units)
{
    int i = blockIdx.x;
    if (i < batch_samples){
    for (int32 f = threadIdx.x; f < units; f += blockDim.x){
        int max0_inp = sequence_len * units;
        int max1_inp = units;
        int max0 = sequence_len * output_depth * units;
        int max1 = output_depth * units;
        int max2 = units;
        for (int j = 0; j < sequence_len; ++j)
        {
          int inp_idx = i * max0_inp + j * max1_inp + f;
          int oup_idx = i * max0 + j * max1 + f;
          output[oup_idx] = inputs[inp_idx];
          indexes[oup_idx] = inp_idx;
        }
        int arg;
        int dilation = 1;
        T val;
        int pad;
        for (int n = 0; n < output_depth - 1; ++n)
        {
          for (int j = 0; j < sequence_len; ++j)
            {
                arg = indexes[i * max0 + j * max1 + n * max2 + f];
                val = output[i * max0 + j * max1 + n * max2 + f];
                pad = (kernel_size - 1) * dilation + 1;
                for (int k = j - pad + 1; k < j; k += dilation)
                {
                  if (k < 0){k = 0;}
                  int test_idx = i * max0 + k * max1 + n * max2 + f;
                  if (output[test_idx] > val)
                  {
                    arg = indexes[test_idx];
                    val = output[test_idx];
                  }
                }
                int oup_idx = i * max0 + j * max1 + (n + 1) * max2 + f;
                output[oup_idx] = val;
                indexes[oup_idx] = arg;
            }
          dilation *= kernel_size;
        }
  }
  }
}

template <typename T>
void DenseKernelLauncher(
        const T* inputs,
        T* output,
        int* indexes,
        const int batch_samples,
        const int sequence_len,
        const int output_depth,
        const int kernel_size,
        const int units)
{
    DenseKernel<T><<<batch_samples, min(THREADS_PER_BLOCK, units)>>>(inputs, output, indexes, batch_samples, sequence_len, output_depth, kernel_size, units);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}

template <typename T>
__global__ void ZeroKernel(T* inputs, const int elements)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += blockDim.x * gridDim.x){
    inputs[index] = (T) 0;
  }  
}

template <typename T>
void ZeroKernelLauncher(T* tensor, const int elements){
     const int blocks = (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
     ZeroKernel<T><<<blocks, THREADS_PER_BLOCK>>>(tensor, elements);
     cudaError_t cudaerr = cudaDeviceSynchronize();
     if (cudaerr != cudaSuccess) printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}

template <typename T>
__global__ void GradKernel(const int* indexes, T* grad_in, const T* grad_out, const int sequence_len, const int output_depth, const int units)
{
    int max0 = sequence_len * output_depth * units;
    int max1 = output_depth * units;
    int max2 = units;
    int i = blockIdx.x;
    for (int32 f = threadIdx.x; f < units; f += blockDim.x){
        for (int j = 0; j < sequence_len; ++j){
            for (int n = 0; n < output_depth; ++n){
            int index = i * max0 + j * max1 + n * max2 + f;
                grad_in[indexes[index]] += grad_out[index];
            }
        }
    }
}
//  for (int32 index = blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += blockDim.x * gridDim.x){
//    CudaAtomicAdd(grad_in + indexes[index], grad_out[index]);
//    //grad_in[indexes[index]] += grad_out[index];
//  }
//}

template <typename T>
void GradKernelLauncher(const int* indexes, T* grad_in, const T* grad_out, const int batch_size, const int sequence_len, const int output_depth, const int units)
{
  GradKernel<T><<<batch_size, min(THREADS_PER_BLOCK, units)>>>(indexes, grad_in, grad_out, sequence_len, output_depth, units);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}  

#define DEFINE_GPU_KERNELS(T)\
   template void DenseKernelLauncher<T>(const T* inputs, T* output, int* indexes, const int batch_samples, const int sequence_len, const int output_depth, const int kernel_size, const int units);\
   template void ZeroKernelLauncher<T>(T* tensor, const int elements);\
   template void GradKernelLauncher<T>(const int* indexes, T* grad_in, const T* grad_out, const int batch_size, const int sequence_len, const int output_depth, const int units);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
