#include "dynamicconv.h"
#include "../ra/ra.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"


using namespace tensorflow;

REGISTER_OP("DynamicConvolution")
  .Attr("T: {float16, float32, float64}")
  .Input("input: T")
  .Input("weight: T")
  .Attr("padding_l: int")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
    c->set_output(0, input_shape);
    return Status::OK();
    });

/*
    Dense Operation GPU
*/

template <typename T>
class DynamicConvOpGPU : public OpKernel {
public:
  int padding_l;
 
  explicit DynamicConvOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("padding_l", &padding_l));
  }

  void Compute(OpKernelContext* context) override {
    // get the input tensor
    const Tensor& input = context->input(0);
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();

    //Check that inputs are three dimensional
    DCHECK_EQ(input_shape.dims(), 3);

    const int miniBatch = input_shape.dim_size(0);
    const int sequenceLength = input_shape.dim_size(2);
    const int numFeatures = input_shape.dim_size(1);

    const Tensor& weight = context->input(1);
    const TensorShape& weight_shape = weight.shape();
    DCHECK_EQ(weight_shape.dims(), 4); // batch, heads, kernel, time
    const int numHeads = weight_shape.dim_size(1);
    const int filterSize = weight_shape.dim_size(2);

    const int numFiltersInBlock = numFeatures / numHeads;

    const dim3 blocks(miniBatch, numFeatures);
    int b_size = nextPowerOf2(filterSize);

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.flat<T>();
    auto weight_tensor = weight.flat<T>();
    auto output_tensor = output->template flat<T>();

    Tensor temp_tensor_input;
    Tensor temp_tensor_filter;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numFeatures * (b_size + filterSize)}, &temp_tensor_input));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numFeatures * b_size * filterSize}, &temp_tensor_filter));
    auto temp_input = temp_tensor_input.flat<T>();
    auto temp_filter = temp_tensor_filter.flat<T>();

    DynamicConvForwardLauncher(input_tensor.data(), // batch, chanels, sequence
			       weight_tensor.data(), // ?, numHeads, filterSize
			       temp_input.data(), // (n_features * batch_size) * (block_dim + fs)
			       temp_filter.data(), // (n_features * batch_size) * n_threads * fs
			       filterSize,
			       padding_l,
			       miniBatch,
			       sequenceLength,
			       numFeatures,
			       numFiltersInBlock,
			       numHeads,
			       output_tensor.data(),
			       blocks,
			       b_size);
  }
};

REGISTER_OP("DynamicConvolutionGrad")
.Attr("T: {float16, float32, float64}")
.Input("grad_output: T")
.Input("input: T")
.Input("weight: T")
.Attr("padding_l: int")
.Output("grad_input: T")
.Output("grad_weight: T");

template <typename T>
class DynamicConvGradOpGPU : public OpKernel {
public:
  int padding_l;
  explicit DynamicConvGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("padding_l", &padding_l));
  }
  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(3, context->num_inputs());

    const Tensor& gradOutput = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& weight = context->input(2);
    
    const TensorShape& input_shape = input.shape();
    const int miniBatch = input_shape.dim_size(0);
    const int sequenceLength = input_shape.dim_size(2);
    const int numFeatures = input_shape.dim_size(1);

    const TensorShape& weight_shape = weight.shape();
    DCHECK_EQ(weight_shape.dims(), 4); // batch, heads, kernel, time
    const int numHeads = weight_shape.dim_size(1);
    const int filterSize = weight_shape.dim_size(2);

    const int numFiltersInBlock = numFeatures / numHeads;
    const int b_size = sequenceLength <= MAX_THREADS_PER_BLOCK ? sequenceLength : 64;
    const int numChunks = int(ceilf(sequenceLength/float(b_size)));
    const dim3 blocks(miniBatch, numHeads, numChunks);

    // create output tensors
    Tensor* gradInput = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &gradInput));
    Tensor* gradWeight = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_shape, &gradWeight));

    // get the Eigen tensors for data access
    auto gradOutputTensor = gradOutput.flat<T>();
    auto inputTensor = input.flat<T>();
    auto weightTensor = weight.flat<T>();

    auto gradInputTensor = gradInput->template flat<T>();
    auto gradWeightTensor = gradWeight->template flat<T>();

    Tensor tempGradOutputFull;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numHeads * numChunks * (b_size + filterSize)}, &tempGradOutputFull));
    auto temp_grad_out = tempGradOutputFull.flat<T>();

    Tensor tempInputFull;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numHeads * numChunks * (b_size + filterSize)}, &tempInputFull));
    auto temp_input = tempInputFull.flat<T>();

    Tensor tempGradSumFull;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numHeads * numChunks * b_size * filterSize}, &tempGradSumFull));
    auto temp_grad_sum = tempGradSumFull.flat<T>();
    
    Tensor bFilterFull;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, {miniBatch * numHeads * numChunks * b_size * filterSize}, &bFilterFull));
    auto temp_b_filter = bFilterFull.flat<T>();

    DynamicConvBackwardLauncher(gradOutputTensor.data(), // B * C * T
				inputTensor.data(), // B * C * T
				weightTensor.data(),
				temp_grad_out.data(), // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
				temp_input.data(), // minibatch, numHeads, numChunks * (blockDim.x + filterSize)
				temp_grad_sum.data(), // minibatch, numHeads, numChunks, num_threads, filterSize
				temp_b_filter.data(), // minibatch, numHeads, numChunks, num_threads, filterSize
				filterSize,
				padding_l,
				miniBatch,
				sequenceLength,
				numFeatures,
				numFiltersInBlock,
				numHeads,
				gradWeightTensor.data(),
				gradInputTensor.data(),
				blocks,
				b_size);
  }
};

#define REGISTER_DYNAMIC_CONV_GPU_KERNELS(T)\
  REGISTER_KERNEL_BUILDER(Name("DynamicConvolution").Device(DEVICE_GPU).TypeConstraint<T>("T"), DynamicConvOpGPU<T>);\
  REGISTER_KERNEL_BUILDER(Name("DynamicConvolutionGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), DynamicConvGradOpGPU <T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_CONV_GPU_KERNELS);

int nextPowerOf2(int n)
{
  if(n >= MAX_THREADS_PER_BLOCK) return MAX_THREADS_PER_BLOCK;
  unsigned int p = 1;
  while (p < n || p < WARP_SIZE)
    p <<= 1;

  return p;
}

