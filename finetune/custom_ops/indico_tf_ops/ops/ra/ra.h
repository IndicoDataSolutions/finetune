#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
template <typename T>
void DenseKernelLauncher(
			 const T* inputs,
			 T* output,
			 int* indexes,
			 const int batch_samples,
			 const int sequence_len,
			 const int output_depth,
			 const int kernel_size,
			 const int units
			 );

template <typename T>
void ZeroKernelLauncher(T* tensor, const int elements);

template <typename T>
void GradKernelLauncher(const int* indexes, T* grad_in, const T* grad_out, const int batch_size, const int sequence_len, const int output_depth, const int units);
