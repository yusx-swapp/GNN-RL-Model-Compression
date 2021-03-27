import os
import argparse

from torch import nn
from torchvision import models
import torch.backends.cudnn as cudnn
from data import resnet
import torch
import logging
import numpy as np

from lib.RL.agent import Memory, Agent

logging.disable(30)
from graph_env.graph_environment import graph_env
from utils.split_dataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR, get_dataset

torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='gnnrl search script')

    # datasets and model
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='ILSVRC', type=str, help='dataset to use (cifar/ILSVRC)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    # parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    # parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--train_size', default=50000, type=int, help='(Fine tuning) training size of the datasets.')
    parser.add_argument('--val_size', default=5000, type=int, help='(Reward caculation) test size of the datasets.')
    parser.add_argument('--f_epochs', default=20, type=int, help='Fast fine-tuning epochs.')

    # pruning

    parser.add_argument('--compression_ratio', default=0.5, type=float,
                        help='compression_ratio')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='n_points_per_layer')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')

    # rl agent
    parser.add_argument('--g_in_size', default=20, type=int, help='initial graph node and edge feature size')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--g_hidden_size', default=50, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--g_embedding_size', default=50, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--hidden_size', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--solved_reward', default=0, type=int, help='stop training if avg_reward > solved_reward')
    parser.add_argument('--log_interval', default=20, type=int, help='print avg reward in the interval')
    parser.add_argument('--max_episodes', default=10000, type=int, help='max training episodes')
    parser.add_argument('--max_timesteps', default=1500, type=int, help='max timesteps in one episode')
    parser.add_argument('--update_timestep', default=100, type=int, help='update policy every n timesteps')
    parser.add_argument('--action_std', default=0.5, type=float, help='constant std for action distribution (Multivariate Normal)')
    parser.add_argument('--K_epochs', default=20, type=int, help='update policy for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float, help='clip parameter for RL')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate for optimizer')

    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')


    parser.add_argument('--log_dir', default='./logs', type=str, help='log dir')
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    # parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    # parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    # parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')

    return parser.parse_args()

def search(env):
    ############## Hyperparameters ##############
    env_name = "gnnrl_search"
    render = False
    solved_reward = args.solved_reward         # stop training if avg_reward > solved_reward
    log_interval = args.log_interval           # print avg reward in the interval
    max_episodes = args.max_episodes        # max training episodes
    max_timesteps = args.max_timesteps        # max timesteps in one episode

    update_timestep = args.update_timestep      # update policy every n timesteps
    action_std = args.action_std            # constant std for action distribution (Multivariate Normal)
    K_epochs = args.K_epochs               # update policy for K epochs
    eps_clip = args.eps_clip              # clip parameter for RL
    gamma = args.gamma                # discount factor

    lr = args.lr                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = args.seed
    #############################################

    state_dim = args.g_in_size
    action_dim = layer_share
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    agent = Agent(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print("Learning rate: ",lr,'\t Betas: ',betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    print("-*"*10,"start search the pruning policies","-*"*10)
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = agent.select_action(state, memory)
            state, reward, done = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                print("-*"*10,"start training the RL agent","-*"*10)
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
                print("-*"*10,"start search the pruning policies","-*"*10)

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if (i_episode % log_interval)!=0 and running_reward/(i_episode % log_interval) > (solved_reward):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './rl_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(agent.policy.state_dict(), './rl_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0




def load_model(model_name,data_root,device=None):
    if device==None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models",'resnet44-014dd654.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet110-1d1ed7c2.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet32-d509ac18.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet20-12fca82f.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "vgg16":
        net = models.vgg16(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    elif model_name == "mobilenetv2":
        from data.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'mobilenetv2.pth.tar')
        print(path)
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)



        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenet":
        from data.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        # net = torch.nn.DataParallel(net)
        sd = torch.load("data/pretrained_models/mobilenet_imagenet.pth.tar")
        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        net.load_state_dict(sd)
        # net = net.cuda()
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenet':
        from data.shufflenet import shufflenet
        net = shufflenet()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetbest.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenetv2':
        from data.shufflenetv2 import shufflenetv2
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

def get_prunable_idx(net,args):
    index = []
    if args.model == 'resnet56':
        for i, module in enumerate(net.modules()):
            #for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                index.append(i)


def get_num_hidden_layer(net,args):
    layer_share=0

    n_layer=0

    if args.model in ['mobilenet']:
        #layer_share = len(list(net.module.features.named_children()))+1

        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    n_layer +=1
                else:
                    n_layer +=1
                    layer_share+=1
    elif args.model in ['mobilenetv2','shufflenet','shufflenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    n_layer +=1
                    layer_share+=1
                else:
                    n_layer +=1

    elif "resnet" in args.model:

        layer_share+=len(list(net.module.layer1.named_children()))
        layer_share+=len(list(net.module.layer2.named_children()))
        layer_share+=len(list(net.module.layer3.named_children()))
        layer_share+=1
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer+=1
    elif args.model == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_share+=1
        n_layer = layer_share
    else:
        raise NotImplementedError
    return n_layer,layer_share

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    device = torch.device(args.device)

    net = load_model(args.model,args.data_root)
    net.to(device)
    cudnn.benchmark = True

    n_layer,layer_share = get_num_hidden_layer(net,args)

    if args.dataset == "ILSVRC":
        path = args.data_root

        train_loader, val_loader, n_class = get_split_valset_ImageNet("ILSVRC", args.data_bsize, 4, args.train_size, args.val_size,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
        input_x = torch.randn([1,3,224,224]).to(device)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")

        train_loader, val_loader, n_class = get_split_train_valset_CIFAR(args.dataset, args.data_bsize, 4, args.train_size, args.val_size,
                                                                         data_root=path, use_real_val=False,
                                                                         shuffle=True)
        input_x = torch.randn([1,3,32,32]).to(device)
    elif args.dataset == "cifar100":
        path = os.path.join(args.data_root, "datasets")

        train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
                                                        data_root=args.data_root)
        input_x = torch.randn([1,3,32,32]).to(device)
    else:
        raise NotImplementedError



    env = graph_env(net,n_layer,args.dataset,val_loader,args.compression_ratio,args.g_in_size,args.log_dir,input_x,device,args)
    search(env)

#python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet110 --compression_ratio 0.4 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset cifar100 --bsize 32 --model shufflenetv2 --compression_ratio 0.2 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --disable graph_encoder --lr_c 0.01 --lr_a 0.01 --dataset cifar10 --bsize 32 --model resnet56 --compression_ratio 0.5 --warmup 1 --pruning_method cp --val_size 1000 --train_episode 200 --log_dir ./logs3
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset ILSVRC --bsize 32 --model mobilenet --compression_ratio 0.25 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs --data_root ../code/data/datasets
