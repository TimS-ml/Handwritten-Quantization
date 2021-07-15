# Overview of the project
ResNet50 + cifar10, post training quantization

## Types of frameworks:
https://jackwish.net/2019/neural-network-quantization-introduction.html
[1] (this proj) Mixed FP32/INT8
- the model itself and input/output are in FP32 format, converts FP32 to INT8 and the reverse
- converts weights to INT8 format
[2] Pure INT8 Inference
- needs to support quantization per operator

## Details
### Quantize Weight
According to NVIDIA and Google, use min-max observer is ok
Because the weight distribution is generally more uniform (I guess)

Per-Channel quantization is good to have (one scale and/or zero_point per slice in the quantized_dimension)

### Quantize Activation
Histogram Observer


### Symmetric vs Asymmetric
- Activations are asymmetric: Many activations are asymmetric in nature and a zero-point is an relatively inexpensive way to effectively get up to an extra binary bit of precision
- Weights are symmetric: forced to have zero-point equal to 0. **Weight values are multiplied by dynamic input and activation values.** This means that there is an unavoidable runtime cost of multiplying the zero-point of the weight with the activation value. By enforcing that zero-point is 0 we can avoid this cost.

=> Tricky part: [What I've learned about neural network quantization](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/)
For -128 to +127: This is inconvenient because there's **one more value on the negative side than the positive, and so requires careful handling if we want to use symmetric ranges** and ensure zero is exactly representable as encoded zero.

## How many bits do we need
According to Stanford CS231N L15 by Song Han, for FC: 2bits, for Conv: 4bits


# Next steps
## Refince the project
Learn from PyTorch:
- Quantization config
- Debug Observer and Placeholder Observer
- Layer fusion (Conv+BN, Conv+BN+ReLU)
  - use scale and zero point **after** ReLU, and map to 0~255
[Quant-Conv-BN-ReLU](https://github.com/Jermmy/pytorch-quantization-demo/blob/e075b87745aa782b6961dba752b3bf06a03e5702/module.py#L288)
[Folding BN ReLU (Chinese)](https://jermmy.github.io/2020/07/19/2020-7-19-network-quantization-4/)


## Further Compress Model (may need re-training) 
Combines Pruning, Quantization, and Huffman encoding into a three stage pipeline that reduces the size of AlexNet by a factor of 35x and VGG-16 by 49x. This results in AlexNet being reduced from 240 to 6.9 MB and VGG-16 from 552 to 11.3 MB.

```
Song Han's four-bit weights require a lookup table, which makes them hard to implement efficiently at runtime
Altering the model architectures so that there's less work to do is usually a much bigger win than tweaking the bit depth
```
Paper: Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding


**Pruning**
One particularly interesting one is weight pruning, where the connections of a network are iteratively removed during training. (Or post-training in some variations.)

Network after Pruning are hard to re-train, consider **Lottery Ticket Hypothesis (sub-network)**, less '0' in weights
Paper: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Weight Sharing**
K-means the value that shared center
For example: [2.09, 2.12, 1.91 ...] => 2
Inference on single number

**Huffman Coding**
Huffman encoding is used in this case to reduce the amount of bits needed to represent the weights in the Quantized Codebook.
If 20 weights map to '2', 10 weights map to '3', and 3 weights map to '8', it would make sense to encode 2 as '00', 3 as '10, and 8 as something like '1110'

**Knowledge Distillation**
Essentially, after the model is trained, a significantly smaller student model is trained to predict the original model.

**SqueezeNet**
[SqueezeNet](https://www.youtube.com/watch?v=ge_RT5wvHvY)

- replace 3x3 filters with 1x1
  - x9 less parameters
- decrease number of input channels to 3x3
- make conv layers large activation maps

**Low rank approximation**
Decompose a conv layers's filters

**Knowledge Distillation**
Paper: (Intel) Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy


## Inference Speed
Focusing on Algorithm level

[Boost Quantization Inference Performance](https://jackwish.net/2019/boost-quant-perf.html)
Pure INT8 Inference
