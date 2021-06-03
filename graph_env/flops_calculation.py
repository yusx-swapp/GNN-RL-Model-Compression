from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np, numpy
from torchvision import models


def layer_flops(layer, input_x):
    output_x = layer.forward(input_x)
    if isinstance(layer, torch.nn.Conv2d):

        c_in = input_x.shape[1]
        c_out = output_x.shape[1]
        h_out = output_x.shape[2]
        w_out = output_x.shape[3]
        kernel_h, kernel_w = layer.kernel_size

        flops = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out / layer.groups

    else:
        raise TypeError
    return flops, output_x

def preserve_flops(Flops,preserve_ratio,model_name,a):
    actions = np.clip(1-a, 0.1, 1)
    flops = deepcopy(Flops)
    if model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(1, len(flops)):
            flops[i] *= preserve_ratio[i - 1]

    elif "mobilenet" == model_name:
        flops[::2] = flops[::2] * (np.array(actions).reshape(-1))
        flops[::2] = flops[::2] * (np.append([1],np.array(actions[:-1]).reshape(-1)).reshape(-1))

        flops[1::2] = flops[1::2] * (np.array(actions[:-1]).reshape(-1))
        flops[1::2] = flops[1::2] * (np.array(actions[:-1]).reshape(-1))

    elif model_name in ['resnet18','resnet50']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(1, len(flops)):
            flops[i] *= preserve_ratio[i - 1]

    elif model_name in ['mobilenetv2']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(0, len(flops)):
            if i+1<len(flops):
                flops[i] *= preserve_ratio[i + 1]
        for i in range(2, len(flops)):
            flops[i] *= preserve_ratio[i - 1]
        for i in range(3, len(flops)):
            if i+1<len(flops):
                flops[i] *= preserve_ratio[i -2 ]
    else:
        raise NotImplementedError
    return flops
def flops_caculation_forward(net, model_name, input_x, preserve_ratio=None):
    # TODO layer flops

    flops = []
    if model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        module = net.module.conv1
        flop, input_x = layer_flops(module, input_x)
        flops.append(flop)

        module_list = [net.module.layer1, net.module.layer2, net.module.layer3]
        for layer in module_list:
            for i, (name, module) in enumerate(layer.named_children()):
                flop, input_x = layer_flops(module.conv1, input_x)
                flops.append(flop)

                flop, input_x = layer_flops(module.conv2, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            # share the pruning index where layers are connected by residual connection
            # flops[0] is the total flops of all the share index layers
            # from share_layers import act_share
            #
            # a_list = act_share(net, a_list,args)

            if len(flops) != len(preserve_ratio):
                raise IndexError

            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]

        flops_share = list(flops[1::2])
        flops_share.insert(0, sum(flops[::2]))
    elif model_name in ['mobilenet','shufflenet','shufflenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0],module.in_channels,input_x.shape[2],input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)


        if preserve_ratio is not None:
            # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)

            if len(flops[::2]) != len(preserve_ratio):
                raise IndexError

            flops[::2] = flops[::2] * np.array(preserve_ratio).reshape(-1)
            flops[::2] = flops[::2] * np.append([1],np.array(preserve_ratio[:-1]).reshape(-1)).reshape(-1)

            flops[1::2] = flops[1::2] * np.array(preserve_ratio[:-1]).reshape(-1)
            flops[1::2] = flops[1::2] * np.array(preserve_ratio[:-1]).reshape(-1)

        flops_share = list(flops[::2])
    elif model_name == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]

        #Here in VGG-16 we dont need to share the pruning index
        flops_share = flops
    elif model_name in ['resnet18','resnet50']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0],module.in_channels,input_x.shape[2],input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):

                flops[i] *= preserve_ratio[i - 1]

        flops_share = flops

    elif model_name in ['mobilenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0],module.in_channels,input_x.shape[2],input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(0, len(flops)):
                if i+1<len(flops):
                    flops[i] *= preserve_ratio[i + 1]
            for i in range(2, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
            for i in range(3, len(flops)):
                if i+1<len(flops):
                    flops[i] *= preserve_ratio[i -2 ]
        flops_share= None
    else:
        raise NotImplementedError

    return flops, flops_share


if __name__ == '__main__':
    net = models.mobilenet_v2()
    print(net)