Resource Management
===================

Finetuning large language models can be very memory and compute intensive. Here are some tips to optimize Finetune if you run into issues.

**GPU Memory**

A machine with a fairly modern gpu (>= 8 GB of memory) is more than enough to run finetune. However, large models can occasionally cause OOM issues,
especially under certain training conditions. Here are some tips to to help diagnose and solve memory problems:

* We have found that very few finetuning tasks benefit from using a large batch size. Reducing batch size to 2-4 will greatly reduce memory use.

    * If you would like to simulate a larger batch size, use the :py:attr:`accum_steps` flag to accumulate gradients over a number of steps before updating
    * Recall that :py:attr:`batch_size` is *per-gpu*. If you are using 3 GPUs with a batch size of 4, you will have an effective batch size of 12.
* Finetune supports gradient checkpointing (recalculating some gradients rather than caching) with the :py:attr:`low_memory_mode` flag, with little noticeable harm to computation speed
* If your dataset contains text that is consistently shorter than 512 tokens, you can lower the model's :py:attr:`max_length` parameter for a large improvement in memory use.
* If you have a very large dataset, ensure that you are not loading the entire file into memory; instead, pass it to a Finetune model with a generator.

**Model Throughput**

Both training and inference are very compute intensive. Here are some tips if you find that your models do not process data as quickly as desired:

* If model speed is very important, consider switching from a transformer-based model to TextCNN or TCN, as they are much faster at the expense of some accuracy.
* If you need to use transformers, consider turning :py:attr:`max_length` down as much as possible, as it will greatly increase your iterations/sec.
* If you need to use multiple models, and find that loading weights/graph compilation takes a long time, check out :doc:`adapter`.
* Keep in mind that model inference is much faster than training, so your deployed model will be able to handle many more examples/sec once it is deployed.



