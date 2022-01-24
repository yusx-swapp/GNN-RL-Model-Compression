import argparse
import os
import torch
from torch import nn
from torch.nn.utils import prune
from torchvision import models

from gnnrl.graph_env.feedback_calculation import top5validate
from gnnrl.graph_env.network_pruning import real_pruning, channel_pruning
from gnnrl.utils.split_dataset import get_dataset



# def boolean_string(s):
#     if s not in {'False', 'True'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'True'
def parse_args():
    parser = argparse.ArgumentParser(description='real pruning')
    #parser.add_argument()
    parser.add_argument('--model', default='vgg16', type=str, help='model to prune')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    parser.add_argument('--n_worker', default=8, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')


    return parser.parse_args()

def load_model(model_name):

    if model_name == "vgg16":
        net = models.vgg16(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from original model..')
            path = os.path.join(args.ckpt_path,'vgg16_20FLOPs_origin.pth')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        net.cuda()

    else:
        raise KeyError
    return net



if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)


    train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
                                                    data_root=args.data_root)

    net = load_model(args.model)
    net.to(device)
    net.cuda()

    for name, module in net.named_modules(): #remove mask
        if isinstance(module, nn.Conv2d):
            module = prune.remove(module,name='weight')

    net = real_pruning(args,net)

    if args.ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from pruned model..')
        path = os.path.join(args.ckpt_path,'vgg16_20FLOPs.pth')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)


    criterion = nn.CrossEntropyLoss().to(device)

    val_top1,val_top5 = top5validate(val_loader, device, net, criterion)

    print( 'Acc1: {:.3f}% | Acc5: {:.3f}%'.format(val_top1,val_top5))




#python gnnrl_real_pruning.py --dataset imagenet --model vgg16 --data_root ../code/data/datasets --ckpt_path data/pretrained_models
