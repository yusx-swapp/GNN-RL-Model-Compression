import sys

from networks.resnet import LambdaLayer

sys.path.append("..")

import torch.nn.utils.prune as prune
import copy
import torch.nn as nn
import torch
import numpy as np

def pruning_cp_fg(net,a_list):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i = 0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            # print("Sparsity ratio",a_list[i])
            prune.ln_structured(module, name='weight', amount=float(1 - a_list[i]), n=2, dim=0)
            i += 1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i += 1
    return newnet

def channel_pruning(net,preserve_ratio):
    '''
    :param net: DNN
    :param preserve_ratio: preserve rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''
    #
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):

            prune.ln_structured(module, name='weight', amount=float(1-preserve_ratio[i]), n=2, dim=0)

            i+=1

    return newnet
def channel_pruning_mobilenet(net,a_list):
    '''
    :param net: DNN
    :param a_list: pruning rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''

    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels:
                continue
            prune.ln_structured(module, name='weight', amount=float(1-a_list[i]), n=2, dim=0)
            i+=1


    return newnet
def unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1 - a_list[i]))
            i += 1

    return newnet
def l1_unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1

    return newnet
def network_pruning(net,a_list):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    candidate_net = channel_pruning(net,a_list)
    # if args.pruning_method == "cp":
    #     candidate_net = channel_pruning(net,a_list)
    # elif args.pruning_method == "fg":
    #     candidate_net = unstructured_pruning(net, a_list)
    # elif args.pruning_method == "cpfg":
    #     candidate_net = pruning_cp_fg(net, a_list)
    # else:
    #     raise KeyError
    return candidate_net

def real_pruning(args,net):
    #IMPORTANT: Before real_pruning, you must finetuning
    '''
        In the previous pruning, we used mask to perform pseudo pruning.
        Here is the real pruning, and return the pruned weights of the module.

        :param weights: module wieghts
        :return: the pruned weights
    '''

    def extract_pruned_weights(pre_preserve_index,weights):
        '''

        :param pre_preserve_index:
        :param weights:
        :return:
        '''

        if pre_preserve_index != None:
            weights = weights[:, pre_preserve_index]

        preserve_index = []
        for i in range(weights.shape[0]):
            w_filter = weights[i]
            if np.where(w_filter != 0)[0].shape[0] == 0:
                continue
            preserve_index.append(i)
        weights = weights[preserve_index]
        return weights, preserve_index
    def real_prune_resblock(module,preserve_index,device):
        w = module.conv1.weight
        weights = w.cpu().detach().numpy()

        new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
        module.conv1.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        module.bn1=torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)


        w = module.conv2.weight
        weights = w.cpu().detach().numpy()

        new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
        module.conv2.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        module.bn2=torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)

        if isinstance(module.shortcut,LambdaLayer):
            in_c1 = module.conv1.weight.data.shape[1]
            out_c2 = module.conv2.weight.data.shape[0]
            pads = out_c2 - in_c1
            # print(out_c2,in_c1)
            # print(module.conv1.weight.data.shape)
            # print(module.conv2.weight.data.shape)

            del module.shortcut
            module.shortcut = LambdaLayer(lambda x:
                            nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pads//2, pads - pads//2), "constant", 0)).to(device)

        return preserve_index
    # for name, module in net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         module = prune.remove(module,name='weight')
    device = torch.device(args.device)

    if "resnet" in args.model:

        conv1 = net.module.conv1
        w = conv1.weight
        weights = w.cpu().detach().numpy()
        new_weights, preserve_index = extract_pruned_weights(None, weights)

        conv1.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        net.module.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)
        for module in net.module.layer1:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.module.layer2:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.module.layer3:
            preserve_index = real_prune_resblock(module, preserve_index, device)
            in_feature = module.conv2.weight.data.shape[0]
        net.module.linear.weight.data = net.module.linear.weight.data[:,:in_feature]
    elif "mobilenet" in args.model:
        preserve_index = None
        for module in net.modules():
            if isinstance(module, nn.Conv2d) and module.groups != module.in_channels:
                w = module.weight

                weights = w.cpu().detach().numpy()
                new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)

                module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
                #module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
                #print(module.weight.data.shape)
                out_c = module.weight.data.shape[0]
            if isinstance(module, nn.BatchNorm2d):
                #print('1asdasdasdasdasdasd')
                w = module.weight
                print(module.weight.data.shape)
                weights = w.cpu().detach().numpy()
                module.weight.data = nn.Parameter(torch.randn(out_c)).to(device)
                print(module.weight.data.shape)
                #module=nn.BatchNorm2d(out_c,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
                module.weight.data = nn.Parameter(torch.randn([out_c,1,3,3])).to(device)
                #module=nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=module.kernel_size,groups=out_c,stride=module.stride,padding=module.padding)
    elif args.model == 'vgg16':
        preserve_index = None
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight
                weights = w.cpu().detach().numpy()
                new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)

                module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
                module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
                #print(module.weight.data.shape)
                out_c = module.weight.data.shape[0]
        net.classifier= nn.Sequential(
            nn.Linear(in_features=out_c*7*7, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        ).to(device)
    return net