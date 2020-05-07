import os

import torch
import torchvision.models as models
from torch import onnx

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
for ii, model in enumerate([wide_resnet50_2]):
    core_name = f'_____{ii}'
    onnx.export(model=model, args=dummy_input,
                f=f"{core_name}.onnx",
                verbose=True,
                input_names=['input'],
                output_names=['output'])

    # mo_run = f'python3 /opt/intel/openvino/deployment_tools/model_optimizer/' \
    #          f'mo.py --input_model {core_name}.onnx --data_type FP16' \
    #          f'--output_dir {os.curdir}'
    mo_run = f'python3 /opt/intel/openvino/deployment_tools/model_optimizer/' \
             f'mo.py --input_model {core_name}.onnx  ' \
             f"--reverse_input_channels " \
             f"--mean_values=[123.675,116.28,103.53] " \
             f'--scale_values=[58.395,57.12,57.375]  ' \
             f'--data_type FP32' \
             f' --output_dir {os.curdir}'
    os.system(mo_run)
# print('\n', mo_run, '\n')

import torch
import torchvision.models as models
from types import FunctionType

#
# def check_first_layer(model):
#     for name, weights in model.named_parameters():
#         w = weights.abs()
#         chn = w.sum(dim=0).sum(-1).sum(-1)
#         # Normalize so that R+G+B=1
#         chn = chn / chn.sum(0).expand_as(chn)
#         chn[torch.isnan(chn)] = 0
#         return chn.detach().numpy()

# models = ['mnasnet1_0']
# chn_info = torch.tensor([])
# for model_name in ['mnasnet1_0']:
#     if model_name[0].islower():
#         attr = getattr(models, model_name)
#         if isinstance(attr, FunctionType):
#             try:
#                 model = attr(pretrained=True)
#                 rgb_vec = check_first_layer(model)
#                 print(f'{model_name: <25} RGB: {rgb_vec}')
#                 rgb_vec = torch.tensor(rgb_vec).view(1, -1)
#                 chn_info = torch.cat((chn_info, rgb_vec), dim=0)
#                 mean = chn_info.mean(dim=0)
#                 std = chn_info.std(dim=0)
#
#                 print(f'         Mean RGB: {mean.numpy()}')
#                 print(f'         STD RGB: {std.numpy()}')
#             except:
#                 pass

