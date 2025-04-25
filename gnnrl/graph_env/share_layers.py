
import torch.nn as nn


def share_layer_index(net,a_list,model_name):
    a_share = []
    if model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        a_share.append(a_list[0])  # 第一层
        i = 1
        for layer in [net.module.layer1, net.module.layer2, net.module.layer3]:
            for name, module in layer.named_children():
                a_share.append(a_list[i])   # 主卷积层
                a_share.append(a_list[0])   # 残差卷积层共享相同动作
                i += 1
        return a_share
    elif model_name in ['mobilenet']:
        i = 0
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    a_share.append(0)
                else:
                    a_share.append(a_list[i])
                    i += 1
    elif model_name in ['mobilenetv2','shufflenet','shufflenetv2']:
        i = 0
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    a_share.append(a_list[i])
                    i += 1
                else:
                    a_share.append(1)

    elif model_name == 'vgg16':
        a_share = a_list
    elif model_name in ['resnet18','resnet50']:
        a_share = a_list
    else:
        a_share = a_list
    return a_share
