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

# %% [markdown] id="TjjTTlkyxfIr"
# # Import

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 619, "status": "ok", "timestamp": 1620739085310, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="tK_s3yloxgYI" outputId="d79a07a8-5052-43af-818a-342498803564"
# %matplotlib inline
# %xmode Verbose
# # %xmode Plain

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26866, "status": "ok", "timestamp": 1620739111569, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="FJvJbxGuxkfL" outputId="19bdb07f-6f1a-4097-c9a2-494c82c013ba"
import os
import sys

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir('/content/drive/My Drive/Project/Quantization/')
    print('Env: colab, run colab init')
    isColab = True
else:
    os.chdir('.')
    cwd = os.getcwd()
    print('Env: local')
    isColab = False

# %% id="XBteJ9JEkWSD"
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

import torchvision

# %% id="a5JEXwXa52i5"
import copy
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


# %% [markdown] id="1jYHtHzbzO7P"
# ## Config

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30307, "status": "ok", "timestamp": 1620739115042, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="lBFH541Wxzys" outputId="73912b6d-e05f-4f40-c2be-23a44380b198"
def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# %% id="f3-jebxkx2QX"
# model save path and prefix
savepath = './checkpoint/' + 'ResNet50_2_'
modelpath = './checkpoint/ResNet50_93.62_44.pt'

kwargs = {'num_workers': 2, 'pin_memory': True}


# %% [markdown] id="FavcFxpDmIwn"
# # Utils

# %% id="l5dVwcQZmKM8"
def calibrate_model(model, loader, device=torch.device('cpu')):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def calibrate_model_n_liter(model, loader, device=torch.device('cpu'), count=3):
    model.to(device)
    model.eval()

    # short calibration
    for inputs, labels in loader:
        if count > 0:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _ = model(inputs)
            count -= 1
        else:
            break


# %% [markdown] id="wlkfcen6zBCs"
# # Data Preprocessing

# %% id="9uDg7261ksZG"
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
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=getdata
    )

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, transform=test_transform, download=getdata
    )

    return input_size, num_classes, train_dataset, test_dataset


# %% id="VY3ARl_0kywt"
input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=128, shuffle=True, **kwargs
# )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, **kwargs
)

# %% id="3YVCcfO0dArp"
temp_data, temp_target = next(iter(test_loader))


# %% [markdown] id="9_8Pwbd7xfPQ"
# # Model

# %% id="v8-icgFAsTkg"
# A modify version of original Pytorch Source Code
# https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/resnet.py

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
        self.conv1 = nn.Conv2d(inplanes, width, 
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               groups=groups,
                               bias=False,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
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


# %% id="gkQgQIV-z6_x"
model = models.resnet._resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, True)

model.conv1 = torch.nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
model.maxpool = torch.nn.Identity()
# model.fc = nn.Sequential(
#     nn.Linear(in_features=2048, out_features=10, bias=True),
#     nn.LogSoftmax(dim=1)
# )

model = model.to(device)

# %% [markdown] id="Mz7HwkjRcKZ3"
# ## Examine Statedict

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40953, "status": "ok", "timestamp": 1620739125739, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="O_u48ID_aLvm" outputId="d5407186-aea6-460d-a203-4e7e65615374"
# checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)

# %% [markdown] id="ZF5Y66MT0-39"
# ### Structure

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40943, "status": "ok", "timestamp": 1620739125740, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="sTIT4sPmathh" outputId="a84f3f90-6e0e-4a61-e69c-750f29eeebb9"
# for m in model.modules():
#     print(m)
#     # if isinstance(m, nn.Conv2d):
#     #     print(m)

print(model)

# %% [markdown] id="wLgaSNUA1ETi"
# ### Param shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40932, "status": "ok", "timestamp": 1620739125740, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="UCee8Bl8cV4R" outputId="7deb0da7-85ca-4826-d815-785d4c4012c2"
# or: for param in model.parameters()
for name, param in model.named_parameters():
    print(name, param.shape)

# %% [markdown] id="ivBPzHOq0Ish"
# ### Conv
#
# Both self.conv1 and self.downsample layers downsample the input when stride != 1
#
# ```python
# if stride != 1 or self.inplanes != planes * block.expansion:
#     downsample = nn.Sequential(
#         conv1x1(self.inplanes, planes * block.expansion, stride),
#         norm_layer(planes * block.expansion),
#     )
# ```

# %% id="B4GCV-eUw7j3"
# # model.conv1?

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 453, "status": "ok", "timestamp": 1620795824342, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="PfNRd5fs0LMi" outputId="6108ae68-e18b-4416-f2e4-479ad13efbee"
print(model.conv1.weight.shape)
print(model.conv1.weight)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41069, "status": "ok", "timestamp": 1620739125906, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="ZTiQ15Zw1vdh" outputId="ee41d6e7-40f0-42e2-dea7-0f344304c3dc"
print(model.layer1[0].conv1.weight.shape)
# print(model.layer1[0].conv1.weight)

# %% [markdown] id="KGzkOu8Y02ZT"
# ### BN

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41059, "status": "ok", "timestamp": 1620739125907, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="g1tVAVmL2Crv" outputId="d08e02f5-d42e-4736-b91a-eac6f116376c"
print(model.bn1.weight.shape)
print(model.bn1.bias.shape)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41162, "status": "ok", "timestamp": 1620739126021, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="WTqscc6pSIzy" outputId="db77bf21-e7d6-4602-b2a6-4ad99a7a6ce3"
print(model.layer4[1].bn1.weight)


# %% [markdown] id="bR31pcuw04w7"
# ### ReLU

# %% id="uS62YKI306Qh"
# print(model.resnet.bn1.bias)

# %% [markdown] id="-0djCXZ3P-gQ"
# # Let's Try Pytorch Built in Quantization Method

# %% id="5WfEGw6Cm2uv"
class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        self.model_fp32 = model_fp32
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 583, "status": "ok", "timestamp": 1620765984368, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="quFcQEPWyg2b" outputId="033cfe9b-13cb-4db4-ae82-0af1c69f14aa"
fused_model = copy.deepcopy(model)

# model.eval()
fused_model.eval()

# %% id="wbgTE8Mr7R_B"
# Fuse the model in place
# model, modules_to_fuse, inplace=False
fused_model = torch.quantization.fuse_modules(fused_model, [['conv1', 'bn1', 'relu']], inplace=True)

for module_name, module in fused_model.named_children():
    if 'layer' in module_name:
        for basic_block_name, basic_block in module.named_children():
            # print(basic_block_name, basic_block)
            torch.quantization.fuse_modules(basic_block, [['conv1', 'bn1', 'relu1'], 
                                                          ['conv2', 'bn2', 'relu2'],
                                                          ['conv3', 'bn3']], inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name == 'downsample':
                    torch.quantization.fuse_modules(sub_block, [['0', '1']], inplace=True)  # cov2d + bn

# %% [markdown] id="Sct1XmizUs4y"
# ## Examine Fused Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41969, "status": "ok", "timestamp": 1620739126878, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="yYIHEmP5Uvkf" outputId="21766ec8-87d7-470e-fe1b-ca6017f6a5b1"
print(fused_model)

# %% id="YLEAVIXWVPf2"
# torch.save(fused_model.state_dict(), savepath + 'fused.pt')
# torch.jit.save(torch.jit.script(fused_model), savepath + 'fused_jit.pt')

# %% id="uxDFnux67OFU"
# Prepare the model for static quantization. 
# This inserts observers in the model that will observe activation tensors during calibration.
quantized_model = QuantizedModel(model_fp32=fused_model)

# config
quantization_config = torch.quantization.get_default_qconfig('fbgemm')  # x86
# quantization_config = torch.quantization.default_qconfig
# quantization_config = torch.quantization.QConfig(
#     activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
#     weight=torch.quantization.MinMaxObserver.with_args(
#         dtype=torch.qint8, 
#         qscheme=torch.per_tensor_symmetric))

quantized_model.qconfig = quantization_config
# print(quantized_model.qconfig)

# %% [markdown] id="3nSQJiR1rvXD"
# ## Examine Model before Calibration
#
# https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 261, "status": "ok", "timestamp": 1620766012486, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="uTZNSNPGkv0-" outputId="1445aa2d-3c26-4d12-ea05-221bc96ad627"
# Prepare the model for static quantization
torch.quantization.prepare(quantized_model, inplace=True)

# %% [markdown] id="3C0TZHJMllcL"
# ## Calibration and Convert

# %% id="RRQY57A_3kpU"
# Calibration!!!
# quantized_model.eval()
# for batch, target in test_loader:
#       model(batch)

# calibrate_model(model=quantized_model, loader=test_loader)
calibrate_model_n_liter(model=quantized_model, loader=test_loader, count=1)

# %% id="4Qz1b_iPEtk-"
# print(quantized_model.model_fp32.conv1)
# torch.save(quantized_model.state_dict(), savepath + 'temp.pt'.format())

# RuntimeError: Hook '_observer_forward_hook' on module 'ConvReLU2d' expected the input argument to be typed as a Tuple but found type: 'Tensor' instead.
# This error occured while scripting the forward hook '_observer_forward_hook' on module ConvReLU2d. 
# If you did not want to script this hook remove it from the original NN module before scripting. 
# This hook was expected to have the following signature: _observer_forward_hook(self, input: Tuple[Tensor], output: Tensor). 
# The type of the output arg is the returned type from either the forward method or the previous hook if it exists. Note that hooks can return anything, but if the hook is on a submodule the outer module is expecting the same return type as the submodule's forward.
# torch.jit.save(torch.jit.script(quantized_model), savepath + 'temp_jit.pt'.format())

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2124, "status": "ok", "timestamp": 1620766065670, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="UwUX8XmkjtRs" outputId="b38a1675-6016-4a40-f263-9d3737fdd3d7"
quantized_model = torch.quantization.convert(quantized_model, inplace=True)

quantized_model.eval()

# Using high-level static quantization wrapper
# The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
# quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

# %% [markdown] id="JgI3mDDzQqYN"
# ## Examine Quantized Model

# %% [markdown] id="3y8QaAQVQxOY"
# ### Structure

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 895445, "status": "ok", "timestamp": 1620653111625, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="QmoNGs6HQfXw" outputId="e6693313-7bca-419b-8b24-11af010532b8"
print(quantized_model)

# %% [markdown] id="shpk3cL-Qzn6"
# ### Param shape

# %% id="bVFRtuidQlG9"
# # # ???
# quantized_model.eval()
# for name, param in quantized_model.named_parameters():
#     print(name, param.shape)

# %% [markdown] id="QBuC1Q8vyl5s"
# # Test Function

# %% id="dwQ011dNykyt"
criterion = nn.CrossEntropyLoss()

def test(model, test_loader, criterion, device='cpu'):
    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            test_loss += criterion(outputs, target).item() * data.size(0)
            correct += torch.sum(preds == target.data)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))

    return test_loss, test_acc


# %% [markdown] id="2EJWa4THytKR"
# ## Test Acc

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 331052, "status": "ok", "timestamp": 1620766415228, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="vlPvtZ2PRMFD" outputId="0334b1cf-e5ba-4874-bab6-74eef1b9df71"
test_loss, test_acc = test(quantized_model, test_loader, criterion)
# print(test_loss, test_acc)

# %% id="vwjAQIt2PhCG"
torch.save(quantized_model.state_dict(), savepath + '{:.2f}_quantized.pt'.format(test_acc))

# The saved module serializes all of the methods, submodules, parameters, and attributes of this module
torch.jit.save(torch.jit.script(quantized_model), savepath + '{:.2f}_quantized_jit.pt'.format(test_acc))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 565825, "status": "ok", "timestamp": 1620691850240, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="RfeDQH3nA6VV" outputId="e576efee-d5b9-47e9-a2e9-fe7a0562398d"
test_loss_fused, test_acc_fused = test(fused_model, test_loader, criterion)
# print(test_loss, test_acc)

# %% [markdown] id="UPIt-N47T_AK"
# # Load Quantized Statedict

# %% [markdown] id="CIQ2VPYaWTpK"
# ## final quantized model
#
# weight + bias + scale + zeropoint

# %% id="Bo7scBtlL6HZ"
# qmodel = torch.jit.load('./checkpoint/ResNet50_93.65_quantized_jit.pt', map_location=device)
qmodel = torch.load('./checkpoint/ResNet50_93.65_quantized.pt', map_location=device)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 170, "status": "ok", "timestamp": 1620690796509, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="I1Zqz6yCUJ2w" outputId="e432c106-c8ca-490a-f22c-ba493f64eff2"
for name in qmodel:
    print(name)
    print(qmodel[name].int_repr())  # change the dtype from qint8 to int8
    break

# %% [markdown] id="lBAccdhsz08g"
# ### Layer
#
# ```
# model_fp32.layer1.0.conv1.weight
# model_fp32.layer1.0.conv1.bias
# model_fp32.layer1.0.conv1.scale
# model_fp32.layer1.0.conv1.zero_point
# ```
#

# %% [markdown] id="VuT3-rtNKUcN"
# ### Scale and Zero_point

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 240, "status": "ok", "timestamp": 1620765812838, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="cI2CrzalKYMh" outputId="aa18b08e-4fb8-4ef0-a4a7-620aaa5a2abb"
print(qmodel['model_fp32.layer1.0.conv1.scale'])
print(qmodel['model_fp32.layer1.0.conv1.zero_point'])

# %% [markdown] id="GdDJneNMV3IC"
# ## fused model after calibration before convert

# %% id="8XnD2N5MF6jF"
temp_model = torch.load('./checkpoint/ResNet50_2_temp.pt', map_location=device)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 220, "status": "ok", "timestamp": 1620765324590, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="yeLv5ARLGAqb" outputId="2c4d6c25-ce1d-4096-881e-860ef22af9ad"
# for name in temp_model:
#     print(name)
#     print(temp_model[name])
#     break

temp_model['model_fp32.conv1.0.weight']

# %% [markdown] id="9qQO7MmxGpga"
# ### Layer

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 839, "status": "ok", "timestamp": 1620764678554, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="HH83cWiuGkaN" outputId="5886bb1b-cec2-46ad-b82e-b07309291226"
for name in temp_model:
    print(name)

# %% [markdown] id="JwBim_MDH3Da"
# ### Histogram

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 231, "status": "ok", "timestamp": 1620765365765, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="k2mU2hDCIB0g" outputId="33534b01-db02-4348-dff4-f2762d42aabc"
print(temp_model['model_fp32.layer1.0.conv2.activation_post_process.histogram'])
print(len(temp_model['model_fp32.layer1.0.conv2.activation_post_process.histogram']))

# %% [markdown] id="51eHTFaeH7nY"
# ### Eps

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 229, "status": "ok", "timestamp": 1620765381741, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="t_Hay0vaIJyJ" outputId="1c6ec7e9-9742-4705-c178-59fb43da67f8"
print(temp_model['model_fp32.layer1.0.conv2.activation_post_process.eps'])
print(len(temp_model['model_fp32.layer1.0.conv2.activation_post_process.eps']))

# %% [markdown] id="N0ziZU8WH9ws"
# ### Min / Max Val

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 268, "status": "ok", "timestamp": 1620765408352, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 240} id="KdrqbRPbITtk" outputId="0e842f3d-8fbc-473e-e678-6a0ba35184cd"
print(temp_model['model_fp32.layer1.0.conv2.activation_post_process.min_val'])
print(temp_model['model_fp32.layer1.0.conv2.activation_post_process.max_val'])
