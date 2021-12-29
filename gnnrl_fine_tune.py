import os
import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import numpy as np
from torch import optim

from torchvision import models

from networks import resnet
from utils.train_utils import accuracy, AverageMeter, progress_bar, get_output_folder
from graph_env.network_pruning import  channel_pruning
from utils.split_dataset import get_dataset
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.reset_parameters()
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

    # for name, module in net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         module = prune.remove(module,name='weight')
    device = torch.device(args.device)


    if "mobilenet" in args.model:
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
        net.module.classifier= nn.Sequential(
            nn.Linear(in_features=out_c*7*7, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        ).to(device)
    return net
def parse_args():
    parser = argparse.ArgumentParser(description='AMC fine-tune script')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--finetuning', action='store_true', help='Finetuning or training')

    return parser.parse_args()


def get_model():

    print('=> Building model..')
    if args.model == 'mobilenet':
        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        if args.finetuning:
            print("Fine-Tuning...")
            net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        #net.apply(weights_init)
        for name,layer in net.named_modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenetv2':
        net = models.mobilenet_v2(pretrained=True)
        if args.finetuning:
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            #path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        # for name,layer in net.named_modules():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenet_0.5flops':
        from networks.mobilenet_cifar100 import MobileNet
        net = MobileNet(n_class=1000, profile='0.5flops')
    elif args.model == 'vgg16':
        net = models.vgg16(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))
        #net = torch.nn.DataParallel(net)
        # if use_cuda and args.n_gpu > 1:
        #     net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'resnet18':
        net = models.resnet18(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'resnet50':
        net = models.resnet50(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet56":
        net = resnet.__dict__['resnet56']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path )

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet44":
        net = resnet.__dict__['resnet44']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet110":
        net = resnet.__dict__['resnet110']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(120, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet32":
        net = resnet.__dict__['resnet32']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(120, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet20":
        net = resnet.__dict__['resnet20']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'shufflenet':
        from networks.shufflenet import shufflenet
        net = shufflenet()
        if args.finetuning:
            print("Finetuning")
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)
    elif args.model == 'shufflenetv2':
        from networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
        if args.finetuning:
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)


    else:
        raise NotImplementedError

    #if use_cuda and args.n_gpu > 1:

    return net.cuda() if use_cuda else net


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))
    # writer.add_scalar('loss/train', losses.avg, epoch)
    # writer.add_scalar('acc/train_top1', top1.avg, epoch)
    # writer.add_scalar('acc/train_top5', top5.avg, epoch)


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        # writer.add_scalar('loss/test', losses.avg, epoch)
        # writer.add_scalar('acc/test_top1', top1.avg, epoch)
        # writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)


def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                    data_root=args.data_root)



    net = get_model()  # real training



    criterion = nn.CrossEntropyLoss()

    print('weight decay  = {}'.format(args.wd))
    print('Using SGD...')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # if args.model == 'mobilenetv2':
    #     print('Using Adam...')
    #     optimizer = Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)
    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model, args.dataset))
        log_dir = get_output_folder('./logs', '{}_{}_finetune'.format(args.model, args.dataset))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        # writer = SummaryWriter(logdir=log_dir)

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader)
            test(epoch, val_loader)

        # writer.close()
        print('=>  best top-1 acc: {}%'.format( best_acc))

'''
python -W ignore gnnrl_fine_tune.py \
    --model=mobilenet \
    --dataset=imagenet\
    --lr=0.005 \
    --n_gpu=1 \
    --batch_size=512 \
    --n_worker=32 \
    --lr_type=cos \
    --n_epoch=200 \
    --wd=4e-5 \
    --seed=2018 \
    --data_root=../code/data/datasets \
    --ckpt_path=logs/mobilenetckpt.best.pth.tar \
    --finetuning 
        
        --eval

    
    --finetuning
'''