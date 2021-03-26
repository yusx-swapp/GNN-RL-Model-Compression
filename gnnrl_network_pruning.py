import os
import argparse
from copy import deepcopy

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
from utils.split_dataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR, \
    get_split_train_valset_CIFAR100, get_dataset

torch.backends.cudnn.deterministic = True
# from lib.agent import DDPG
from utils.train_utils import get_output_folder, to_numpy, to_tensor, plot_learning_curve


def parse_args():
    parser = argparse.ArgumentParser(description='gnnrl search script')
    #parser.add_argument()
    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    #graph encoder
    parser.add_argument('--node_feature_size', default=50, type=int, help='the initial node feature size')
    parser.add_argument('--pool_strategy', default='mean', type=str, help='pool strategy(mean/diff), defualt:mean')
    parser.add_argument('--embedding_size', default=30, type=int, help='embedding size of DNN\'s hidden layers')

    # datasets and model
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='ILSVRC', type=str, help='dataset to use (cifar/ILSVRC)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--train_size', default=50000, type=int, help='(Fine tuning) training size of the datasets.')
    parser.add_argument('--val_size', default=5000, type=int, help='(Reward caculation) test size of the datasets.')
    parser.add_argument('--f_epochs', default=20, type=int, help='Fast fine-tuning epochs.')

    # pruning

    parser.add_argument('--compression_ratio', default=0.5, type=float,
                        help='compression_ratio')
    parser.add_argument('--pruning_method', default='cp', type=str,
                        help='method to prune (fg/cp/cpfg for fine-grained and channel pruning)')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='n_points_per_layer')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    # ddpg

    parser.add_argument('--g_in_size', default=20, type=int, help='initial graph node and edge feature size')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--g_hidden_size', default=50, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--g_embedding_size', default=50, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--hidden_size', default=300, type=int, help='hidden num of first fully connect layer')

    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=25, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.99, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--disable', default=None, type=str, help='cuda/cpu')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # export

    parser.add_argument('--log_dir', default='./logs', type=str, help='log dir')
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')

    return parser.parse_args()

def search(env):
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for RL
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    state_dim = args.g_in_size
    # action_dim = env.action_space.shape[0]
    action_dim = layer_share
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    agent = Agent(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

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
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(agent.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


def train(agent, env, output,args):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < args.train_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(observation, episode=episode)
        #print(action)
        # env response with next_observation, reward, terminate_info
        observation2, reward, done = env.step(action)
        #observation2 = (observation2)


        T.append([reward, observation, observation2, action, done])

        if episode % int(args.train_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = observation2

        if done:  # end of episode
            print("Search Episode: ",episode)


            final_reward = T[-1][0]
            print('final_reward: {}\n'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []
            print('-' * 40)


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
    if args.pruning_method == "cp":
        prunable_layer_types = [torch.nn.Conv2d]
    else:
        prunable_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
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

        train_loader, val_loader, n_class = get_split_valset_ImageNet("ILSVRC", 128, 4, args.train_size, args.val_size,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
        input_x = torch.randn([1,3,224,224]).to(device)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")

        train_loader, val_loader, n_class = get_split_train_valset_CIFAR('cifar10', 256, 4, args.train_size, args.val_size,
                                                                         data_root=path, use_real_val=False,
                                                                         shuffle=True)
        input_x = torch.randn([1,3,32,32]).to(device)
    elif args.dataset == "cifar100":
        path = os.path.join(args.data_root, "datasets")

        # train_loader, val_loader, n_class = get_split_train_valset_CIFAR100('cifar100', 256, 4, args.train_size, args.val_size,
        #                                                                  data_root=path, use_real_val=True,
        #                                                                  shuffle=True)
        train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
                                                        data_root=args.data_root)
        input_x = torch.randn([1,3,32,32]).to(device)
    else:
        raise NotImplementedError



    env = graph_env(net,n_layer,args.dataset,val_loader,args.compression_ratio,args.g_in_size,args.log_dir,input_x,device,args)
    # hierarchical_graph = env.model_to_graph()
    # agent = Agent(layer_share,args.g_in_size)
    # agent.cuda()
    #agent.load_weights(args.log_dir)
    #env._set_graph_encoder(agent.graph_encoder)
    # if args.disable == "graph_encoder":
    #     agent.graph_grad_status(False)
    #
    search(env)
    # search(agent, env, args.output, args)

#python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet110 --compression_ratio 0.4 --pruning_method cp --train_episode 100 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset cifar100 --bsize 32 --model shufflenetv2 --compression_ratio 0.2 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --disable graph_encoder --lr_c 0.01 --lr_a 0.01 --dataset cifar10 --bsize 32 --model resnet56 --compression_ratio 0.5 --warmup 1 --pruning_method cp --val_size 1000 --train_episode 200 --log_dir ./logs3
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset ILSVRC --bsize 32 --model mobilenet --compression_ratio 0.25 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs --data_root ../code/data/datasets
