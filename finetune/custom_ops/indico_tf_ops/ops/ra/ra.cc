#include "ra.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"


using namespace tensorflow;

/*
    Register Dense operation
*/

REGISTER_OP("RecursiveAgg")
  .Attr("T: {float16, float32, float64}")
  .Input("input: T")
  .Attr("kernel_size: int")
  .Attr("pool_len: int")
  .Attr("output_depth: int")
  .Output("output: T")
  .Output("index: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));

    shape_inference::DimensionHandle samples = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle length = c->Dim(input_shape, 1);
    shape_inference::DimensionHandle units = c->Dim(input_shape, 2);

    int32 output_depth;
    TF_RETURN_IF_ERROR(c->GetAttr("output_depth", &output_depth));
    shape_inference::DimensionHandle depth_dim = c->MakeDim(output_depth);
    c->set_output(0, c->MakeShape({samples, length, depth_dim, units}));
    c->set_output(1, c->MakeShape({samples, length, depth_dim, units}));
    return Status::OK();
  });

/*
    Dense Operation CPU
*/

template <typename T>
class DenseOpCPU : public OpKernel {
public:
  int32 kernel_size;
  int32 pool_len;
  int32 output_depth;
  explicit DenseOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context, context->GetAttr("pool_len", &pool_len));
    OP_REQUIRES_OK(context, context->GetAttr("output_depth", &output_depth));
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseOpCPU\n");

    // get the input tensor
    const Tensor& input = context->input(0);
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    
    //Check that inputs are three dimensional
    DCHECK_EQ(input_shape.dims(), 3);
    
    const int batch_samples = input_shape.dim_size(0);
    //printf("batch_samples %d\n", batch_samples);

    const int seq_length = input_shape.dim_size(1);
    //printf("input_feature_width %d\n", input_feature_width);

    const int units  = input_shape.dim_size(2);
    //printf("units %d\n", units);

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(seq_length);
    output_shape.AddDim(output_depth);
    output_shape.AddDim(units);

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    Tensor* index = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &index));
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.flat<T>();
    auto output_tensor = output->flat<T>();
    auto output_index = index->flat<int32>();
    
    int32 max0_inp = seq_length * units;
    int32 max1_inp = units;
    int32 max0 = seq_length * output_depth * units;
    int32 max1 = output_depth * units;
    int32 max2 = units;
    for (int i = 0; i < batch_samples; ++i)
    {
      for (int j = 0; j < seq_length; ++j)
      {
        for (int f = 0; f < units; ++f){
          int64 inp_idx = i * max0_inp + j * max1_inp + f;
          int64 oup_idx = i * max0 + j * max1 + f;
          output_tensor(oup_idx) = input_tensor(inp_idx);
          output_index(oup_idx) = inp_idx;
        }
      }
    }
    int32 arg;
    int32 dilation = 1;
    T val;
    int32 pad;
    for (int n = 0; n < output_depth - 1; ++n)
    {
      for (int i = 0; i < batch_samples; ++i)
      {
        for (int j = 0; j < seq_length; ++j)
        {
          for (int f = 0; f < units; ++f)
          {
            arg = output_index(i * max0 + j * max1 + n * max2 + f);
            val = output_tensor(i * max0 + j * max1 + n * max2 + f);
            pad = (kernel_size - 1) * dilation + 1;
            for (int k = j - pad + 1; k < j; k += dilation)
            {
              if (k < 0){k = 0;}
              int64 test_idx = i * max0 + k * max1 + n * max2 + f;
              if (output_tensor(test_idx) > val)
              {
                arg = output_index(test_idx);
                val = output_tensor(test_idx);
              }
            }
            int32 oup_idx = i * max0 + j * max1 + (n + 1) * max2 + f;
            output_tensor(oup_idx) = val;
            output_index(oup_idx) = arg;
          }
        }
      }
      dilation *= kernel_size;
    }
  }
};


/*
    Dense Operation GPU
*/

template <typename T>
class DenseOpGPU : public OpKernel {
public:
  int kernel_size;
  int pool_len;
  int output_depth;
  explicit DenseOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context, context->GetAttr("pool_len", &pool_len));
    OP_REQUIRES_OK(context, context->GetAttr("output_depth", &output_depth));
  }

  void Compute(OpKernelContext* context) override {
    //printf("DenseOpCPU\n");

    // get the input tensor
    const Tensor& input = context->input(0);
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();

    //Check that inputs are three dimensional
    DCHECK_EQ(input_shape.dims(), 3);

    const int batch_samples = input_shape.dim_size(0);
    //printf("batch_samples %d\n", batch_samples);

    const int seq_length = input_shape.dim_size(1);
    //printf("input_feature_width %d\n", input_feature_width);

    const int units  = input_shape.dim_size(2);
    //printf("units %d\n", units);

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(seq_length);
    output_shape.AddDim(output_depth);
    output_shape.AddDim(units);

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    Tensor* index = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &index));
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.flat<T>();
    auto output_tensor = output->template flat<T>();
    auto output_index = index->template flat<int32>();

    DenseKernelLauncher<T>(
        input_tensor.data(),
        output_tensor.data(),
        output_index.data(),
        batch_samples,
        seq_length,
        output_depth,
        kernel_size,
        units
     );
  }
};


/*
    DenseGrad Operation CPU
*/

REGISTER_OP("RecursiveAggGrad")
   .Attr("T: {float16, float32, float64}")
  .Input("grad: T")
  .Input("input: T")
  .Input("argmax: int32")
  .Output("grad_input: T");


template <typename T>
class DenseGradOpCPU : public OpKernel {
public:
  explicit DenseGradOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseGradOpCPU\n");
    DCHECK_EQ(3, context->num_inputs());

    const Tensor& grad = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& amax = context->input(2);

    TensorShape grad_shape = grad.shape();
    TensorShape input_shape = input.shape();
    TensorShape amax_shape = amax.shape();

    // create output tensors
    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

    // get the Eigen tensors for data access
    auto grad_tensor = grad.flat<T>();
    auto input_tensor = input.flat<T>();
    auto amax_tensor = amax.flat<int32>();
    
    auto grad_input_tensor = grad_input->flat<T>();

    int iter_dim = amax_shape.dim_size(0) * amax_shape.dim_size(1) * amax_shape.dim_size(2) * amax_shape.dim_size(3);  //Number of values in each sample
    int input_dim = input_shape.dim_size(0) * input_shape.dim_size(1) * input_shape.dim_size(2);


    for (int64 i = 0; i < input_dim; ++i)
    {
        grad_input_tensor(i) = (T) 0.0;
    }

    for (int64 i = 0; i < iter_dim; ++i)
    {
        grad_input_tensor(amax_tensor(i)) +=  grad_tensor(i);
    }
  }
};



/*
    DenseGrad Operation GPU
*/

void InputGradKernelLauncher(const double* grads, const double* weights, const int input_feature_width, const int batch_samples, const int units, double* grad_inputs);
void WeightsGradKernelLauncher(const double* grads, const double* inputs, const int input_feature_width, const int batch_samples, const int units, double* grad_weights);
void BiasesGradKernelLauncher(const double* grads, const int input_feature_width, const int batch_samples, const int units, double* grad_biases);


template <typename T>
class DenseGradOpGPU : public OpKernel {
public:
  explicit DenseGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  void Compute(OpKernelContext* context) override {
    //printf("DenseGradOpCPU\n");
    DCHECK_EQ(3, context->num_inputs());

    const Tensor& grad = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& amax = context->input(2);

    TensorShape grad_shape = grad.shape();
    TensorShape input_shape = input.shape();
    TensorShape amax_shape = amax.shape();

    // create output tensors
    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

    // get the Eigen tensors for data access
    auto grad_tensor = grad.flat<T>();
    auto input_tensor = input.flat<T>();
    auto amax_tensor = amax.flat<int32>();

    auto grad_input_tensor = grad_input->flat<T>();

    int iter_dim = amax_shape.dim_size(0) * amax_shape.dim_size(1) * amax_shape.dim_size(2) * amax_shape.dim_size(3);  //Number of values in each sample
    int input_dim = input_shape.dim_size(0) * input_shape.dim_size(1) * input_shape.dim_size(2);

    ZeroKernelLauncher<T>(grad_input_tensor.data(), input_dim);
    GradKernelLauncher<T>(amax_tensor.data(), grad_input_tensor.data(), grad_tensor.data(), amax_shape.dim_size(0), amax_shape.dim_size(1), amax_shape.dim_size(2), amax_shape.dim_size(3));
  }
};

#define REGISTER_RA_GPU_KERNELS(T)\
    REGISTER_KERNEL_BUILDER(Name("RecursiveAgg").Device(DEVICE_GPU).TypeConstraint<T>("T"), DenseOpGPU<T>);\
    REGISTER_KERNEL_BUILDER(Name("RecursiveAggGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), DenseGradOpGPU<T>);
#define REGISTER_RA_CPU_KERNELS(T)\
    REGISTER_KERNEL_BUILDER(Name("RecursiveAggGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), DenseGradOpCPU<T>); \
    REGISTER_KERNEL_BUILDER(Name("RecursiveAgg").Device(DEVICE_CPU).TypeConstraint<T>("T"), DenseOpCPU<T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_RA_GPU_KERNELS);
//TF_CALL_REAL_NUMBER_TYPES(REGISTER_RA_CPU_KERNELS);



