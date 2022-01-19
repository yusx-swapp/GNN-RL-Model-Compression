
import torch.nn as nn


def share_layer_index(net,a_list,model_name):
    a_share = []
    if model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        #share the pruning index where layers are connected by residual connection

        a_share.append(a_list[0])
        i=1
        for name, module in net.module.layer1.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
        for name, module in net.module.layer2.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
        for name, module in net.module.layer3.named_children():
            a_share.append(a_list[i])
            a_share.append(a_list[0])
            i+=1
    elif model_name in ['mobilenet']:
        # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)
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
        #Here in VGG-16 we dont need to share the pruning index
        a_share = a_list
    elif model_name in ['resnet18','resnet50']:
        a_share = a_list
    else:
        a_share = a_list
    return a_share
