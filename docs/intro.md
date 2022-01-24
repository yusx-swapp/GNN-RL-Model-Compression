
## Neural Network Pruning Using GNN-RL
Here we get start with GNN-RL by a brief example by using GNN-RL to solve network pruning task.
Below are steps:

1. load target neural network    
2. define GNN-RL environment
3. define GNN-RL reinforcement learning agent
4. search an training the RL agent for pruning policy


## Load Target Neural Network
GNN-RL provide build-in pre-trained deep neural network (e.g ResNet-20/32-56), can be easily load by ``gnnrl.load_networks.load_model`` .
    
```python
from gnnrl.utils.load_networks import load_model
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet56', data_root='.')
net = net.to(device)
```

Iterate the network and get the number of convolutional layers (convolutional layers are ready to be pruned by GNN-RL),
```python
from torch import nn
n_layer = 0
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        n_layer +=1
```
Or you can use GNN-RL build-in function ``gnnrl.utils.net_info.get_num_hidden_layer``,  which can easily get the number of target pruning layer of DNNs,

```python
from gnnrl.utils.net_info import get_num_hidden_layer
n_layer,_ = get_num_hidden_layer(net,'resnet56')
```

Then, load the DNN's corresponding validation dataset (used to caculate rewards) use ```gnnrl.data.getdataset```. Here we load the cifar-10 dataset with batch size 256.
```python
from gnnrl.utils.split_dataset import get_dataset
train_loader, val_loadtrain_loader, val_loader, n_class = get_dataset("cifar10", batch_size=256, n_worker=8,data_root='.')
```
## DNN-Graph Environment for Neural Network Pruning
GNN-RL modeling the DNN as a computatuinal graph, and constructs a graph environment to simulation the topology change when pruning. Environment can be constructed by ```gnnrl.graph_env```
```python
from gnnrl.graph_env.graph_environment import graph_env
input_x = torch.randn([1,3,32,32]).to(device)
env = graph_env(net,n_layer,"cifar10",val_loader,compression_ratio=0.4,g_in_size=20,log_dir=".",input_x=input_x,max_timesteps=5,model_name="resnet56",device=device)
```
Here the `input_x` is a random place holder for a single data point in the chosen dataset, and used to caculate the FLOPs or MACs.


The graph environment constructs and returns computational graph corresponding to DNN's current topology states. Meanwhile, the graph environment evaluates the size of DNN (e.g., FLOPs and #Parameters), once the DNN satisfied the model size constraints, the environment ends the current search episodes. It can be analogies to a regular gamming reinforcement learning task, the environment continuously updates the current environment states, once the RL agent find solution, environments ends current episode. 
<!-- *Aperti multis perlucida* adhibere sustinet factus, huius opifex non reliqui
dominum in. Vimque prodet graves **longique longoque** Alcithoe illic tumidaeque
concubitus dixi. Nox dentes iram quacumque parte crurumque patrem, at formosus
limite. Sine ipse [bovem](http://www.habenasdixit.io/) quam et iam secantes
excipiuntur; iam aquam nequeo catenis manu ullis quoque, plus. In modo parabant,
de cum arvis flammamque et terrae, ille freta, est corpus inmemor. -->
## Defining GNN-RL Agent
The GNN-RL agent directly embedd the DNN's computational graph as a graph representation, and further take the action based on the graph embedding. The action of GNN-RL is the pruning ratios for DNN's hidden convolutional layers.
```python
from gnnrl.lib.RL.agent import Agent
betas = (0.9, 0.999)
agent = Agent(state_dim=20, action_dim=n_layer, action_std = 0.5, lr = 0.0003, betas = (0.9, 0.999), gamma = 0.99, K_epochs = 10, eps_clip = 0.2)
```

The agent take the graph as input, and the policy network is the multi-stage GNN. The agent updates through DDPG reinforcement learning process.

## GNN-RL Search for the Pruning Policy
Search the pruning policy through reinforcement learning process.
First, define the search hyper-parameters,
```python
import numpy as np
env_name = "gnnrl_search"
render = False
solved_reward = -10  # stop training if avg_reward > solved_reward
log_interval = 50  # print avg reward in the interval
max_episodes = 100  # max training episodes (search rounds)
max_timesteps = 5  # max timesteps in one episode
update_timestep = 50  # update policy every n timesteps
betas = (0.9, 0.999)
random_seed = 0

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
# logging variables
running_reward = 0
avg_length = 0
time_step = 0
```
Create RL memory for update the policy,
```python
from gnnrl.lib.RL.agent import Memory
memory = Memory()
```
Start RL search iteration,    

```python
print("-*" * 10, "start search the pruning policies", "-*" * 10)
# training loop
for i_episode in range(1, max_episodes + 1):
    state = env.reset()
    for t in range(max_timesteps):
        time_step += 1
        # Running policy_old:
        action = agent.select_action(state, memory)
        state, reward, done = env.step(action, t + 1)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if time_step % update_timestep == 0:

            print("-*" * 10, "start training the RL agent", "-*" * 10)
            agent.update(memory)
            memory.clear_memory()
            time_step = 0

            print("-*" * 10, "start search the pruning policies", "-*" * 10)

        running_reward += reward
        if render:
            env.render()
        if done:
            break
    avg_length += t

```
Logging and save the intermediate state dictionary,

```python
    # stop training if avg_reward > solved_reward
    if (i_episode % log_interval) != 0 and running_reward / (i_episode % log_interval) > (solved_reward):
        print("Solved!")
        torch.save(agent.policy.state_dict(), './rl_solved_{}.pth'.format(env_name))
        break;

    # save every 500 episodes
    if i_episode % 500 == 0:
        torch.save(agent.policy.state_dict(), './'  + '_rl_{}.pth'.format(env_name))
        torch.save(agent.policy.actor.graph_encoder.state_dict(),
                   './'+'_rl_graph_encoder_actor_{}.pth'.format(env_name))
        torch.save(agent.policy.critic.graph_encoder_critic.state_dict(),
                   './' +  '_rl_graph_encoder_critic_{}.pth'.format(env_name))
    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))
    
        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
```


GNN-RL provides easy way to search and train the policy network, by calling the ```gnnrl.search```, you only need define the search hyper-parameters:

```python
from gnnrl.search import search
search(env,agent, update_timestep=100,max_timesteps=5, max_episodes=15000,
       log_interval=10, solved_reward=-10, random_seed=None)
```

<!-- 
## Multi-Stage Graph Neural Network

Feremus iamque silvarum parce: in iam pars aura volucrem eripuit. Plangore et
merui ubi carebat contra; cumulo illa Hymettia illo sed!

## Reinforcement Learning Task Definition

Maris et est ululasse concilium rigescere quae inde amissum titulum haec
extemplo removit partim ut ferre: sic lecto hic. Trifida fata. Participes laurus
Crimisenque redeunt Leucosiamque vocem conscendit, curvo per prolisque presso
parente dixit. Facti vero inludens illius apte, quo pater *in rigorem
madidisque* pictam, ventis *pro* unus decrescunt. Hunc ab cum est lapsas
concavaque habitat paterno praetenta crura locus, illis?

> Finierat sinus favet undis fecit flexerat habeoque. Innixa ipse odium, licuit,
> [stellatus](http://arma.net/) parantem solvere celerique socia.

Avertitur Paraetonium caput. [Auro loco](http://frustra-cum.org/laborum)
aequora? -->
