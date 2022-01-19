import time



import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top5validate(val_loader, device, model, criterion):
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            val_top1.update(acc1.item(), input.size(0))
            val_top5.update(acc5.item(), input.size(0))

            # measure elapsed time

    return val_top1.avg, val_top5.avg





# def error_caculation(model, val_loader, device, finetune=False, train_loader=None):
#     criterion = nn.CrossEntropyLoss().to(device)
#
#     val_top1, val_top5 = top5validate(val_loader, device, model, criterion)
#     return val_top1, val_top5


def reward_caculation(pruned_model, val_loader, device):
    # here we do the pseudo pruning, by zero mask those pruned channel


    criterion = nn.CrossEntropyLoss().to(device)
    val_top1, val_top5 = top5validate(val_loader, device, pruned_model, criterion)
    #val_top1, val_top5 = error_caculation(pruned_model, val_loader, device)
    acc = val_top1


    error = 100 - acc
    reward = error * -1

    tp5reward = val_top5-100
    # return float(reward), float(acc)
    return float(reward),float(acc),float(tp5reward),float(val_top5)


def validate(val_loader, device, model, criterion):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # if args.half:
            #     input_var = input_var.half()

            # compute output
            output = model(input_var)
            # print(output.shape,target_var.shape)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

'''

def RewardCaculation_ImageNet(args,a_list,n_layers,DNN,best_accuracy,train_loader,val_loader,device,root = './logs'):
    if len(a_list) < n_layers:
        return 0

    new_net = pruning_cp_fg(DNN,a_list)
    #new_net = unstructured_pruning(DNN,a_list)
    #new_net=l1_unstructured_pruning(DNN,a_list)

    new_net,_, acc = ErrorCaculation_ImageNet(new_net,train_loader,val_loader,device)

    error = 100-acc
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(new_net.state_dict(), root+'/'+args.model+'.pkl')
        f = open(root+"/action_list.txt", "w")
        print(a_list)
        for line in a_list:
            f.write(str(line))
            f.write('\n')
        f.close()
    print("best accuracy",best_accuracy)
    reward = error*-1
    return float(reward),float(best_accuracy)

def RewardCaculation_CIFAR(args,a_list,n_layers,DNN,best_accuracy,train_loader,val_loader,root = './logs'):
    if len(a_list) < n_layers:
        return 0

    new_net = network_pruning(DNN,a_list,args)
    device = torch.device(args.device)
    new_net,_, acc = ErrorCaculation_FineTune(new_net,train_loader,val_loader,device)

    error = 100-acc
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(new_net.state_dict(), root+'/'+args.model+'.pkl')
        f = open(root+"/action_list.txt", "w")

        for line in a_list:
            f.write(str(line))
            f.write('\n')
        f.close()
    print("Best Accuracy of Compressed Model ",best_accuracy)
    reward = error*-1
    return float(reward),float(best_accuracy)



def error_caculation_finetune(model,train_loader,val_loader,device):
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Fine Tuning ...................")
    
    
    model, top1, top5 = train_model_top5(model, dataloaders, criterion, optimizer_ft, device, num_epochs=30, is_inception=False)
    return model, top1, top5



def ErrorCaculation_CIFAR(model,val_loader,device,root='../DNN/datasets'):
    cudnn.benchmark = True


    # evaluate on validation set
    criterion = nn.CrossEntropyLoss().to(device)
    acc = validate(val_loader,device, model, criterion)

        #model = torch.load('model.pkl')
    return 100-acc


'''

'''
def FlopsCaculation(DNN,H,W):
    H_in, W_in = H, W
    Flops = []
    for name, module in DNN.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            C_in = module.in_channels
            C_out = module.out_channels
            stride_h,stride_w = module.stride
            kernel_h,kernel_w = module.kernel_size
            padding = module.padding
            if padding != (0,0):
                H_out = H_in
                W_out = W_in
            else:
                H_out = (H_in - kernel_h) / stride_h + 1
                W_out = (W_in - kernel_w) / stride_w + 1
            Flop = H_out * W_out * (C_in * (2 * kernel_h * kernel_w - 1) + 1) * C_out
            Flops.append(Flop)
            H_in = H_out
            W_in = W_out


    return Flops
    
    
    def flops_caculation(DNN,h,w,a_list=None):
    # TODO: use module forwad to compute H_out and W_out will be more general
    h_in, w_in = h, w
    Flops = []
    if a_list==None:
        for name, module in DNN.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                c_in = module.in_channels
                c_out = module.out_channels
                stride_h, stride_w = module.stride
                kernel_h, kernel_w = module.kernel_size
                padding = module.padding

                h_out = int((h_in +2 * padding[0]-kernel_h)/stride_h +1)
                w_out = int((w_in +2 * padding[0]-kernel_w)/stride_w +1)

                Flop = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out
                Flops.append(Flop)
                h_in = h_out
                w_in = w_out
    else:
        i=0
        for name, module in DNN.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                c_in = module.in_channels
                if i > 0:
                    c_in = np.floor((a_list[i-1]) * c_in)
                c_out = module.out_channels
                c_out = np.floor((a_list[i]) * c_out)

                stride_h,stride_w = module.stride
                kernel_h,kernel_w = module.kernel_size
                padding = module.padding

                h_out = int((h_in + 2 * padding[0] - kernel_h) / stride_h + 1)
                w_out = int((w_in + 2 * padding[0] - kernel_w) / stride_w + 1)

                Flop = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out
                Flops.append(Flop)
                h_in = h_out
                w_in = w_out
                i+=1


    return Flops

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


def flops_caculation_forward(net, args, input_x, a_list=None):
    # TODO layer flops

    flops = []
    if 'resnet' in args.model:
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

        if a_list is not None:
            # share the pruning index where layers are connected by residual connection
            # flops[0] is the total flops of all the share index layers
            from share_layers import act_share

            a_list = act_share(net, a_list,args)

            if len(flops) != len(a_list):
                raise IndexError

            flops = flops * np.array(a_list).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= a_list[i - 1]

        flops_share = list(flops[1::2])
        flops_share.insert(0, sum(flops[::2]))
    elif args.model == 'mobilenet':
        for module in net.module.conv1:
            if isinstance(module,torch.nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        for i, (blocks) in enumerate(net.module.features.children()):
            for name,module in blocks.named_children():
                if isinstance(module, torch.nn.Conv2d):
                    flop, input_x = layer_flops(module, input_x)
                    flops.append(flop)


        if a_list is not None:
            # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)

            if len(flops[::2]) != len(a_list):
                raise IndexError

            flops[::2] = flops[::2] * np.array(a_list).reshape(-1)
            flops[1::2] = flops[1::2] * np.array(a_list[:-1]).reshape(-1)


        flops_share = list(flops[::2])

    elif args.model == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if a_list is not None:
            flops = flops * np.array(a_list).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= a_list[i - 1]

        #Here in VGG-16 we dont need to share the pruning index
        flops_share = flops

    else:
        raise NotImplementedError

    return flops, flops_share

'''
