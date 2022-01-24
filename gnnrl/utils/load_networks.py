import os

import torch
from torchvision import models

from gnnrl.networks import resnet


def load_model(model_name,data_root=None,device=None):

    package_directory = os.path.dirname(os.path.abspath(__file__))
    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10','resnet56.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks',  "pretrained_models",'cifar10','resnet44.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet110.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet32.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet20.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name =='resnet18':
        net = models.resnet18(pretrained=True)
        net = torch.nn.DataParallel(net)

    elif model_name =='resnet50':
        net = models.resnet50(pretrained=True)
        net = torch.nn.DataParallel(net)

    elif model_name == "vgg16":
        net = models.vgg16(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    elif model_name == "mobilenetv2":
        net = models.mobilenet_v2(pretrained=True)
        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenet":
        from gnnrl.networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        # net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'mobilenet_imagenet.pth.tar')
        sd = torch.load(path)

        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        net.load_state_dict(sd)
        # net = net.cuda()
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenet':
        from gnnrl.networks.shufflenet import shufflenet
        net = shufflenet()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetbest.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenetv2':
        from gnnrl.networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetv2.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)

    else:
        raise KeyError
    return net
