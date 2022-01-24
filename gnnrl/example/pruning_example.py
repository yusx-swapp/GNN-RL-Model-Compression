from gnnrl.utils.load_networks import load_model
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet56','./data')
net = net.to(device)

from torch import nn
n_layer = 0
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        n_layer +=1


from gnnrl.utils.net_info import get_num_hidden_layer
n_layer,_ = get_num_hidden_layer(net,'resnet56')



from gnnrl.utils.split_dataset import get_dataset
train_loader, val_loader, n_class = get_dataset("cifar10", batch_size=256, n_worker=8,data_root=' /work/LAS/jannesar-lab/yusx/data')




from gnnrl.graph_env.graph_environment import graph_env
input_x = torch.randn([1,3,32,32]).to(device)
env = graph_env(net,n_layer,"cifar10",val_loader,compression_ratio=0.4,g_in_size=20,log_dir=".",input_x=input_x,max_timesteps=5,model_name="resnet56",device=device)


from gnnrl.lib.RL.agent import Agent
betas = (0.9, 0.999)
agent = Agent(state_dim=20, action_dim=n_layer, action_std = 0.5, lr = 0.0003, betas = (0.9, 0.999), gamma = 0.99, K_epochs = 10, eps_clip = 0.2)

from gnnrl.search import search
search(env,agent, update_timestep=100,max_timesteps=5, max_episodes=15000,
       log_interval=10, solved_reward=-10, random_seed=None)