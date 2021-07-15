# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import packages

# %% colab={"base_uri": "https://localhost:8080/"} id="qI0osRmY9yo_" outputId="b41ad9bd-a7ee-4120-a5a0-0160286db251"
# %matplotlib inline
# %xmode Verbose
# # %xmode Plain

# %% colab={"base_uri": "https://localhost:8080/"} id="FJvJbxGuxkfL" outputId="9fac5905-cdd9-4608-df8d-50dc87774a1e"
import os
import sys

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir('/content/drive/My Drive/Project/Quantization/')  # replace to your google drive path
    print('Env: colab, run colab init')
    isColab = True
else:
    os.chdir('.')
    cwd = os.getcwd()
    print('Env: local')
    isColab = False

# %% id="uEz6aTd2NOIQ"
import copy

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from torchvision import datasets, transforms
from typing import Type, Callable, Union, List, Optional
# from tqdm import tqdm

# %% [markdown] id="1jYHtHzbzO7P"
# ## Config

# %% colab={"base_uri": "https://localhost:8080/"} id="lBFH541Wxzys" outputId="f6677ae4-c590-4d21-f1f9-5dfa4ae2104e"
# Since currently we are using fp32, theoretically it suports cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# for reproduce
def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# %% id="f3-jebxkx2QX"
quant_activation_bits = 4
quant_weight_bits = 4

# model save path and prefix
savepath = './checkpoint/' + 'ResNet50_2_'
modelpath = './checkpoint/ResNet50_93.62_44.pt'  # my pre-trained ResNet50 on GPU
save_final_model = False  # save final quantized model or not

# for data loader
# kwargs = {'num_workers': 2, 'pin_memory': True}
kwargs = {'num_workers': 2}

# %% [markdown] id="ktrYu011Nlik"
# # ResNet 50
#
# This model can be quantized using PyTorch built-in method. Modified from
#
# https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/resnet.py

# %% [markdown] id="q0pS3aXGKHaf"
# ## Bottleneck


# %% id="Xv9XlX78KJtb"
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes,
                               width,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width,
                               width,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               groups=groups,
                               bias=False,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width,
                               planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.float_add = nn.quantized.FloatFunctional()
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.float_add.add(identity, out)
        out = self.relu3(out)

        return out


# %% [markdown] id="TDsQtFQGKRg8"
# ## ResNet-50


# %% id="9_LKJtme95FA"
class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3,
        #                        self.inplanes,
        #                        kernel_size=7,
        #                        stride=2,
        #                        padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,
                                      0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: Type[Union[Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# %% id="OfSGlW_WIsz2"
model = ResNet(Bottleneck, [3, 4, 6, 3])
model = model.to(device)

# %% colab={"base_uri": "https://localhost:8080/"} id="Jy3wHvTPJQ3X" outputId="e0e54121-29ac-4c9e-a3cf-98fdd310f5a3"
# The model is trained on cuda, use `map_location` to load it into CPU
checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)

# %% [markdown] id="XBm6sxwLYQ0U"
# # Observer Base


# %% id="JRODoK0bIJqZ"
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'Layer':
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'Channel':
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]  # output tensors having 1 fewer dimension than input
            max_val = torch.max(input, 1)[0]
        elif self.q_level == 'FC':  # for linear channel
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)


# %% [markdown] id="mVsrXQjQ_rsc"
# ## MinMax Observer
#
# \begin{array}{ll}
# x_\text{min} &= \begin{cases}
#     \min(X) & \text{if~}x_\text{min} = \text{None} \\
#     \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
# \end{cases}\\
# x_\text{max} &= \begin{cases}
#     \max(X) & \text{if~}x_\text{max} = \text{None} \\
#     \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
# \end{cases}\\
# \end{array}
#
#
# \begin{aligned}
#     \text{if Symmetric:}&\\
#     &s = 2 \max(|x_\text{min}|, x_\text{max}) /
#         \left( Q_\text{max} - Q_\text{min} \right) \\
#     &z = \begin{cases}
#         0 & \text{if dtype is qint8} \\
#         128 & \text{otherwise}
#     \end{cases}\\
#     \text{Otherwise:}&\\
#         &s = \left( x_\text{max} - x_\text{min}  \right ) /
#             \left( Q_\text{max} - Q_\text{min} \right ) \\
#         &z = Q_\text{min} - \text{round}(x_\text{min} / s)
# \end{aligned}
#


# %% id="qndlEMhFUGtv"
# ObserverBase contains a default `forward` function
class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'Layer':
            self.min_val = torch.zeros((1), dtype=torch.float32)
            self.max_val = torch.zeros((1), dtype=torch.float32)
        elif self.q_level == 'Channel':
            self.min_val = torch.zeros((out_channels, 1, 1, 1),
                                       dtype=torch.float32)
            self.max_val = torch.zeros((out_channels, 1, 1, 1),
                                       dtype=torch.float32)
        elif self.q_level == 'FC':
            self.min_val = torch.zeros((out_channels, 1), dtype=torch.float32)
            self.max_val = torch.zeros((out_channels, 1), dtype=torch.float32)

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'Channel':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


# %% [markdown] id="Ff1VR9VyIGJz"
# ## Histogram Observer
#
# Modified form PyTorch source code. The scale and zero point are computed as follows:
#
# 1. Create the histogram of the incoming inputs.
#     - The histogram is computed continuously,
#     - and the ranges per bin change with every new tensor observed.
#
# 2. Search the distribution in the histogram for optimal min/max values.
#     - The search for the min/max values ensures the minimization of the quantization error with respect to the floating point model.


# %% id="pm_DrY7VIejj"
class HistogramObserver(nn.Module):
    def __init__(self, q_level, out_channels=None, dst_nbins=8):
        super(HistogramObserver, self).__init__()
        self.num_flag = 0
        self.q_level = q_level
        self.out_channels = out_channels
        self.num_flag = 0
        self.min_val = torch.zeros((1), dtype=torch.float32)
        self.max_val = torch.zeros((1), dtype=torch.float32)
        self.bins = 2048
        self.histogram = torch.zeros(self.bins)
        self.dst_nbins = dst_nbins

    # norm = density * (end^3 - begin^3) / 3
    def _get_norm(self, delta_begin, delta_end, density):
        norm = (delta_end * delta_end * delta_end -
                delta_begin * delta_begin * delta_begin) / 3
        return density * norm

    # Compute the quantization error if we use start_bin to end_bin as the
    # min and max to do the quantization.
    def compute_quantization_error(self, next_start_bin, next_end_bin):
        bin_width = (self.max_val - self.min_val) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin +
                                     1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        # [1, 2, 3, ...]
        src_bin = torch.arange(self.bins)

        # `distances` from the beginning of first dst_bin to 
        #   the beginning and end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(src_bin_begin // dst_bin_width, 0,
                                       self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(src_bin_end // dst_bin_width, 0,
                                     self.dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins)

        # norm += d(delta)
        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins) * delta_end, 
                               density)

        # norm += d(dst)
        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * \
                    self._get_norm(torch.tensor(-dst_bin_width / 2), 
                                   torch.tensor(dst_bin_width / 2),
                                   density)

        dst_bin_of_end_center = (dst_bin_of_end * dst_bin_width +
                                 dst_bin_width / 2)
        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center

        # norm += d(new delta)
        norm += self._get_norm(torch.tensor(delta_begin), 
                               delta_end, 
                               density)

        return norm.sum().item()

    def non_linear_param_search(self):
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self.compute_quantization_error(next_start_bin,
                                                   next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

    @torch.no_grad()
    def forward(self, input):
        min_val = torch.min(input)
        max_val = torch.max(input)
        self.update_range(min_val, max_val)
        
        # Generate histogram
        torch.histc(input, self.bins, out=self.histogram)

        new_min, new_max = self.non_linear_param_search()
        self.update_range(new_min, new_max)


# %% [markdown]
# ## Fake Histogram Observer 

# %%
# Accuracy it's a Percentile Observer
# Since the goal of PyTorch HistogramObserver is to remove outlier, 
#   so simply choose the specific percentile of a histogram
class FakeHistogramObserver(nn.Module):
    def __init__(self, q_level, momentum=0.1, out_channels=None, hist_percentile=0.9999):
        super(FakeHistogramObserver, self).__init__()
        self.momentum = momentum
        self.hist_percentile = hist_percentile
        self.num_flag = 0
        self.q_level = q_level
        self.out_channels = out_channels
        self.min_val = torch.zeros((1), dtype=torch.float32)
        self.max_val = torch.zeros((1), dtype=torch.float32)

    @torch.no_grad()
    def forward(self, input):
        # input, k, dim
        max_val_cur = torch.kthvalue(input.abs().view(-1),
                                     int(self.hist_percentile *
                                         input.view(-1).size(0)),
                                     dim=0)[0]

        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val \
                + self.momentum * max_val_cur
        self.max_val.copy_(max_val)

# %% [markdown] id="D9qEFXtwkqMW"
# # Default Quantizer
#
# Clamp all elements in input into the range [ min, max ]
#
# https://pytorch.org/docs/stable/generated/torch.clamp.html
#


# %% id="U4QTbVlLIyWv"
# This is for QAT, for PTQ use torch.round() is ok
# # class Round(Function):
# #     @staticmethod
# #     def forward(self, input):
# #         output = torch.round(input)
# #         return output
# 
# #     @staticmethod
# #     def backward(self, grad_output):
# #         grad_input = grad_output.clone()
# #         return grad_input

# Symmetric
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
# `register_buffer` should not to be considered a model parameter
class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag):
        super(Quantizer, self).__init__()

        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        # scale/zero_point/eps
        if self.observer.q_level == 'Layer':
            self.register_buffer(
                'scale', 
                torch.ones((1), 
                            dtype=torch.float32))
            self.register_buffer(
                'zero_point',
                torch.zeros((1), 
                            dtype=torch.float32))
        elif self.observer.q_level == 'Channel':
            self.register_buffer(
                'scale',
                torch.ones((self.observer.out_channels, 1, 1, 1),
                           dtype=torch.float32))
            self.register_buffer(
                'zero_point',
                torch.zeros((self.observer.out_channels, 1, 1, 1),
                            dtype=torch.float32))
        elif self.observer.q_level == 'FC':
            self.register_buffer(
                'scale',
                torch.ones((self.observer.out_channels, 1),
                           dtype=torch.float32))
            self.register_buffer(
                'zero_point',
                torch.zeros((self.observer.out_channels, 1),
                            dtype=torch.float32))
        self.eps = torch.tensor((torch.finfo(torch.float32).eps),
                                dtype=torch.float32)  # eps(1.1921e-07)

        if self.activation_weight_flag == 0:  # weight
            self.quant_min_val = torch.tensor((-((1 << (self.bits - 1)) - 1)))
            self.quant_max_val = torch.tensor(((1 << (self.bits - 1)) - 1))
        elif self.activation_weight_flag == 1:  # activation
            self.quant_min_val = torch.tensor((-((1 << (self.bits - 1)) - 1)))
            self.quant_max_val = torch.tensor(((1 << (self.bits - 1)) - 1))
        else:
            print('activation_weight_flag error')

    def update_qparams(self):
        quant_range = float(self.quant_max_val -
                            self.quant_min_val) / 2  # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val),
                                torch.abs(
                                    self.observer.max_val))  # since symmetric, we need max val
        self.scale = float_range / quant_range  # scale
        self.scale = torch.max(self.scale,
                               self.eps)  # processing for very small scale
        self.zero_point = torch.zeros_like(self.scale)  # zero_point

    # def round(self, input):
    #     output = Round.apply(input)
    #     return output

    def forward(self, input):
        if self.training:
            self.observer(input)
            self.update_qparams()  # update scale and zero_point
        # Quantize and DeQuantize
        # [1] Quantized value clamp to [quant_min_val, quant_max_val]
        # [2] Round to int
        # [3] DeQuantize to float
        # output = (torch.clamp(self.round(input / self.scale - self.zero_point),
        #                       self.quant_min_val, self.quant_max_val) \
        #             + self.zero_point) * self.scale
        _output = torch.round(input / self.scale - self.zero_point)
        output = (torch.clamp(_output,
                              min=int(self.quant_min_val),
                              max=int(self.quant_max_val)) \
                        + self.zero_point) * self.scale

        return output


# %% [markdown] id="dKTLMebymnbF"
# # Quantize Layers

# %% [markdown] id="TbNcXcdZYq6i"
# ## QAdaAvgPooling


# %% id="1QSc5j74YqVW"
class QAdaAvgPooling2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, a_bits=8, hist_percentile=0.9999):
        super(QAdaAvgPooling2d, self).__init__(output_size)
        self.activation_quantizer = Quantizer(
            bits=a_bits,
            observer=FakeHistogramObserver(q_level='Layer',
                                           hist_percentile=hist_percentile),
            # observer=HistogramObserver(q_level='Layer',
            #                            dst_nbins=a_bits),
            activation_weight_flag=1)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(quant_input, self.output_size)
        return output


# %% [markdown] id="eFxe8WLARV3v"
# ## QReLU


# %% id="PFa4ptyMRXwO"
class QReLU(nn.ReLU):
    def __init__(self, inplace=False, a_bits=8, hist_percentile=0.9999):
        super(QReLU, self).__init__(inplace)
        self.activation_quantizer = Quantizer(
            bits=a_bits,
            observer=FakeHistogramObserver(q_level='Layer',
                                           hist_percentile=hist_percentile),
            # observer=HistogramObserver(q_level='Layer',
            #                            dst_nbins=a_bits),
            activation_weight_flag=1)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output


# %% [markdown] id="HkhOdss8Ym3N"
# ## QLinear


# %% id="eF8evdfvYon_"
# For ResNet50, nn.Linear is for FC layers
class QLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 direct_inference=False,
                 hist_percentile=0.9999):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.direct_inference = direct_inference
        self.activation_quantizer = Quantizer(
            bits=a_bits,
            observer=FakeHistogramObserver(q_level='Layer',
                                           hist_percentile=hist_percentile),
            # observer=HistogramObserver(q_level='Layer',
            #                            dst_nbins=a_bits),
            activation_weight_flag=1)
        self.weight_quantizer = Quantizer(bits=w_bits,
                                          observer=MinMaxObserver(
                                              q_level='FC',
                                              out_channels=out_features),
                                          activation_weight_flag=0)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.direct_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


# %% [markdown] id="sDpbKb3BWCYM"
# ## QConv


# %% id="MLu-f0iPJaUV"
class QConv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 direct_inference=False,
                 hist_percentile=0.9999):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.direct_inference = direct_inference
        self.activation_quantizer = Quantizer(
            bits=a_bits,
            observer=FakeHistogramObserver(q_level='Layer',
                                           hist_percentile=hist_percentile),
            # observer=HistogramObserver(q_level='Layer',
            #                            dst_nbins=a_bits),
            activation_weight_flag=1)
        self.weight_quantizer = Quantizer(bits=w_bits,
                                          observer=MinMaxObserver(
                                              q_level='Layer', out_channels=None),
                                          activation_weight_flag=0)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.direct_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


# %% [markdown] id="lJymgOi3PkCk"
# # Prepare

# %% id="1Q0phpjgfXxp"
def Quantize_layer_prepare(module,
                           a_bits=8,
                           w_bits=8,
                           direct_inference=False,
                           hist_percentile=0.9999):

    for name, child in module.named_children():
        # [1] Conv2d
        if isinstance(child, nn.Conv2d):
            quant_conv = QConv(child.in_channels,
                               child.out_channels,
                               child.kernel_size,
                               stride=child.stride,
                               padding=child.padding,
                               dilation=child.dilation,
                               groups=child.groups,
                               bias=False,
                               a_bits=a_bits,
                               w_bits=w_bits,
                               direct_inference=direct_inference,
                               hist_percentile=hist_percentile)
            quant_conv.weight.data = child.weight
            module._modules[name] = quant_conv

        # [2] Linear
        elif isinstance(child, nn.Linear):
            quant_linear = QLinear(child.in_features,
                                   child.out_features,
                                   bias=True,
                                   a_bits=a_bits,
                                   w_bits=w_bits,
                                   direct_inference=direct_inference,
                                   hist_percentile=hist_percentile)
            quant_linear.bias.data = child.bias
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear

        # [3] ReLU
        elif isinstance(child, nn.ReLU):
            quant_relu = QReLU(inplace=child.inplace,
                               a_bits=a_bits,
                               hist_percentile=hist_percentile)
            module._modules[name] = quant_relu

        # [4] AdaptiveAvgPool2d, that is what we use in resnet
        # https://discuss.pytorch.org/t/adaptive-avg-pool2d-vs-avg-pool2d/27011
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            quant_adaptive_avg_pool = QAdaAvgPooling2d(
                output_size=child.output_size,
                a_bits=a_bits,
                hist_percentile=hist_percentile)
            module._modules[name] = quant_adaptive_avg_pool

        # [5] Go deeper to the Child
        else:
            Quantize_layer_prepare(child,
                                   a_bits=a_bits,
                                   w_bits=w_bits,
                                   direct_inference=direct_inference,
                                   hist_percentile=hist_percentile)


# %% colab={"base_uri": "https://localhost:8080/"} id="m1DFDMe2Pl9D" outputId="09b6f707-3b99-452d-9c4a-9e12d3ef6a33"
fused_model = copy.deepcopy(model)
fused_model.eval()

print()

# %% colab={"base_uri": "https://localhost:8080/"} id="gRv7cxlrUMTe" outputId="54aef915-1c5a-4f3a-d30e-2ab6c04518fb"
Quantize_layer_prepare(fused_model, a_bits=quant_activation_bits, w_bits=quant_weight_bits)

# %% colab={"base_uri": "https://localhost:8080/"} id="CBoflvrAYW-j" outputId="e91add9c-aa3c-4c79-fbd0-d2bcf2f9f426"
# The model is quantized and wait for calibration
for name, param in fused_model.named_parameters():
    print(name, param.shape)

# %% [markdown] id="aE8-uGAbfpUS"
# # Main


# %% id="9uDg7261ksZG"
# get data
def get_CIFAR10(getdata=False):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     transform=train_transform,
                                     download=getdata)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    transform=test_transform,
                                    download=getdata)

    return input_size, num_classes, train_dataset, test_dataset


# %% id="VY3ARl_0kywt"
input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=128,
                                           shuffle=True,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          **kwargs)

# %% [markdown] id="hm-0PVSuX-O7"
# # Calib


# %% colab={"base_uri": "https://localhost:8080/"} id="On9QGK86X9mJ" outputId="e2ab4789-56aa-4510-e32f-81e34554b82e"
def calib_model_n_liter(batch_num_liter=50):
    fused_model.train()

    batch_num = 0
    for data, _ in train_loader:
        _ = fused_model(data)

        batch_num += 1
        if batch_num > batch_num_liter:
            break
        if batch_num % 5 == 0:
            print('Batch:', batch_num)
    return


def test():
    fused_model.eval()
    fused_model.to(device)

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            outputs = fused_model(data)
            _, preds = torch.max(outputs, 1)

            test_loss += criterion(outputs, target).item() * data.size(0)
            correct += torch.sum(preds == target.data)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))

    return test_loss, test_acc


criterion = nn.CrossEntropyLoss()

calib_model_n_liter()
test_loss, test_acc = test()

# %% id="gVMlHxCrq0FA"
if save_final_model:
    torch.save(fused_model.state_dict(),
               savepath + '{}a-{}w-bit_{:.2f}.pt'.format(
                                                     quant_activation_bits,
                                                     quant_weight_bits,
                                                     test_acc))
