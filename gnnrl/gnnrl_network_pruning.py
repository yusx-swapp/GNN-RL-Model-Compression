import os

import torch.backends.cudnn as cudnn

import torch
import logging
import numpy as np

from gnnrl_search import search
from lib.RL.agent import Memory, Agent
from parameter import parse_args
from utils.load_networks import load_model
from utils.net_info import get_num_hidden_layer

logging.disable(30)
from graph_env.graph_environment import graph_env
from utils.split_dataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR, get_dataset

torch.backends.cudnn.deterministic = True



# times = []
# def search(env):
#
#     ############## Hyperparameters ##############
#     env_name = "gnnrl_search"
#     render = False
#     solved_reward = args.solved_reward         # stop training if avg_reward > solved_reward
#     log_interval = args.log_interval           # print avg reward in the interval
#     max_episodes = args.max_episodes        # max training episodes
#     max_timesteps = args.max_timesteps        # max timesteps in one episode
#
#     update_timestep = args.update_timestep      # update policy every n timesteps
#     action_std = args.action_std            # constant std for action distribution (Multivariate Normal)
#     K_epochs = args.K_epochs               # update policy for K epochs
#     eps_clip = args.eps_clip              # clip parameter for RL
#     gamma = args.gamma                # discount factor
#
#     lr = args.lr                 # parameters for Adam optimizer
#     betas = (0.9, 0.999)
#
#     random_seed = args.seed
#     #############################################
#
#     state_dim = args.g_in_size
#     action_dim = layer_share
#     if random_seed:
#         print("Random Seed: {}".format(random_seed))
#         torch.manual_seed(random_seed)
#         env.seed(random_seed)
#         np.random.seed(random_seed)
#
#     memory = Memory()
#     agent = Agent(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
#
#     if args.transfer:
#         print("#Start Topology Transfer#")
#         critc_state = torch.load("resnet56_rl_graph_encoder_critic_gnnrl_search.pth",device)
#         actor_state = torch.load("resnet56_rl_graph_encoder_actor_gnnrl_search.pth",device)
#         agent.policy.critic.graph_encoder_critic.load_state_dict(critc_state)
#         agent.policy.actor.graph_encoder.load_state_dict(actor_state)
#         for param in agent.policy.actor.graph_encoder.parameters():
#             param.requires_grad = False
#         for param in agent.policy.critic.graph_encoder_critic.parameters():
#             param.requires_grad = False
#         print("#Successfully Load Pretrained Graph Encoder!#")
#
#     print("Learning rate: ",lr,'\t Betas: ',betas)
#
#
#
#
#     # logging variables
#     running_reward = 0
#     avg_length = 0
#     time_step = 0
#
#     print("-*"*10,"start search the pruning policies","-*"*10)
#     # training loop
#     for i_episode in range(1, max_episodes+1):
#         state = env.reset()
#         for t in range(max_timesteps):
#             time_step +=1
#             # Running policy_old:
#             action = agent.select_action(state, memory)
#             state, reward, done = env.step(action,t+1)
#
#
#
#
#             # Saving reward and is_terminals:
#             memory.rewards.append(reward)
#             memory.is_terminals.append(done)
#
#             # update if its time
#             if time_step % update_timestep == 0:
#                 # start = time.time()
#
#                 print("-*"*10,"start training the RL agent","-*"*10)
#                 agent.update(memory)
#                 memory.clear_memory()
#                 time_step = 0
#
#                 # end = time.time()
#                 # times.append(end-start)
#
#                 print("-*"*10,"start search the pruning policies","-*"*10)
#
#
#             running_reward += reward
#             if render:
#                 env.render()
#             if done:
#                 break
#
#         avg_length += t
#
#         # stop training if avg_reward > solved_reward
#         if (i_episode % log_interval)!=0 and running_reward/(i_episode % log_interval) > (solved_reward):
#             print("########## Solved! ##########")
#             torch.save(agent.policy.state_dict(), './rl_solved_{}.pth'.format(env_name))
#             break
#
#         # save every 500 episodes
#         if i_episode % 500 == 0:
#             torch.save(agent.policy.state_dict(), './'+args.model+'_rl_{}.pth'.format(env_name))
#             torch.save(agent.policy.actor.graph_encoder.state_dict(),'./'+args.model+'_rl_graph_encoder_actor_{}.pth'.format(env_name))
#             torch.save(agent.policy.critic.graph_encoder_critic.state_dict(),'./'+args.model+'_rl_graph_encoder_critic_{}.pth'.format(env_name))
#         # logging
#         if i_episode % log_interval == 0:
#             avg_length = int(avg_length/log_interval)
#             running_reward = int((running_reward/log_interval))
#
#             print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
#             avg_reward.append(running_reward)
#             running_reward = 0
#             avg_length = 0
#             print(avg_reward)




if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    net = load_model(args.model,args.data_root,device)
    net.to(device)
    cudnn.benchmark = True

    n_layer,layer_share = get_num_hidden_layer(net,args.model)

    if args.dataset == "imagenet":
        path = args.data_root

        train_loader, val_loader, n_class = get_split_valset_ImageNet("imagenet", args.data_bsize, args.n_worker, args.train_size, args.val_size,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
        input_x = torch.randn([1,3,224,224]).to(device)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")

        train_loader, val_loader, n_class = get_split_train_valset_CIFAR(args.dataset, args.data_bsize, args.n_worker, args.train_size, args.val_size,
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



    env = graph_env(net,n_layer,args.dataset,val_loader,args.compression_ratio,args.g_in_size,args.log_dir,input_x,args.max_timesteps,args.model,device)
    betas = (0.9, 0.999)
    agent = Agent(state_dim=args.g_in_size, action_dim=layer_share, action_std = args.action_std, lr = args.lr, betas = betas, gamma = args.gamma, K_epochs = args.K_epochs, eps_clip = args.eps_clip)

    # search(env)
    search(env,agent, update_timestep=args.update_timestep,max_timesteps=args.max_timesteps, max_episodes=args.max_episodes,
           log_interval=10, solved_reward=args.solved_reward, random_seed=args.seed)
#python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.4 --log_dir ./logs --val_size 5000
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset cifar100 --bsize 32 --model shufflenetv2 --compression_ratio 0.2 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --dataset imagenet --model mobilenet --compression_ratio 0.2 --val_size 5000  --log_dir ./logs --data_root ../code/data/datasets
#python -W ignore gnnrl_network_pruning.py --dataset imagenet --model resnet18 --compression_ratio 0.2 --val_size 5000  --log_dir ./logs --data_root ../code/data/datasets
