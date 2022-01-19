


## Reinforcement learning task definition
To prune a given DNN, the user provides the model size constraint (e.g., FLOPs-constraint). The DNN-Graph environment receives the constraint, takes the DNN’s hierarchical computational graph as the environment state, and leverages the GNN-RL agent to search for a compression policy.
To model the pruning as RL task, we need define the following:

1. Environment states and space
2. Action space
3. Reward function
4. Search constraints

The environment states are the generated hierarchical computational graph. The actions made by the RL agent are pruning ratios for hidden layers, which within a continuous space. The reward function is the compressed DNN's Top-1 error on the validation set. The search constraints is the FLOPs-constraints, once the compressed model satisfy the constraints, the environment end current search episode.


## Policy neural network
Various RL policies aim to search within a continuous action space, such as proximal policy optimization (PPO) and deep deterministic policy gradient (DDPG). GNN-RL opted for the PPO RL policy, as it provided much better performance.
### GNN-RL defualt policy network
GNN-RL defualt policy network ```gnnrl.lib.RL.agent```.

    from gnnrl.lib.RL.agent import Agent
    state_dim = 30  #computational graph node feature size 
    action_dim = 20 #number of hidden layers to be pruned
    agent = Agent(state_dim, action_dim)

### Define your own policy network

Define an actor network:

    class ActorNetwork(nn.Module):
        def __init__(self,g_in_size, g_hidden_size, g_embedding_size,hidden_size, nb_actions,
                    chkpt_dir='tmp/rl'):
            super(ActorNetwork, self).__init__()

            self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_rl')
            self.graph_encoder = multi_stage_graph_encoder(g_in_size, g_hidden_size, g_embedding_size)
            self.linear1 = nn.Linear(g_embedding_size,hidden_size)
            self.linear2 = nn.Linear(hidden_size,nb_actions)
            self.nb_actions = nb_actions
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()

            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.to(self.device)

        def forward(self, state):
            g = self.graph_encoder(state)
            actions = self.relu(self.linear1(g))
            actions = self.tanh(self.linear2(actions))
            return actions

        def save_checkpoint(self):
            T.save(self.state_dict(), self.checkpoint_file)

        def load_checkpoint(self):
            self.load_state_dict(T.load(self.checkpoint_file))

Define a critic network:

    class CriticNetwork(nn.Module):
        def __init__(self, g_in_size, g_hidden_size, g_embedding_size,
                    chkpt_dir='tmp/rl'):
            super(CriticNetwork, self).__init__()

            self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_rl')
            self.graph_encoder_critic = multi_stage_graph_encoder(g_in_size, g_hidden_size, g_embedding_size)
            self.linear1 = nn.Linear(g_embedding_size, 1)
            self.tanh = nn.Tanh()
            # self.sigmoid = nn.Sigmoid()

            # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.to(self.device)

        def forward(self, state):
            g = self.graph_encoder_critic(state)
            value = self.tanh(self.linear1(g))
            return value

        def save_checkpoint(self):
            T.save(self.state_dict(), self.checkpoint_file)

        def load_checkpoint(self):
            self.load_state_dict(T.load(self.checkpoint_file))

Define the actor-critic network:

    class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim, action_std):
            super(ActorCritic, self).__init__()
            # action mean range -1 to 1
            actor_cfg = {
                'g_in_size':state_dim,
                'g_hidden_size':50,
                'g_embedding_size':50,
                'hidden_size':200,
                'nb_actions':action_dim,

            }
            critic_cfg = {

                'g_in_size':state_dim,
                'g_hidden_size':50,
                'g_embedding_size':50,
            }
            self.actor = ActorNetwork(**actor_cfg)
            self.critic = CriticNetwork(**critic_cfg)

            self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
            
        def forward(self):
            raise NotImplementedError
        
        def act(self, state, memory):
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).to(device)
            
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            
            return action.detach()
        
        def evaluate(self, state, action):   
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            
            dist = MultivariateNormal(action_mean, cov_mat)
            
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_value = self.critic(state)
            
            return action_logprobs, torch.squeeze(state_value), dist_entropy


## DNN-graph reinforcemente learning environment 
![graph_env](../images/graph_env.png)
Figure shows the DNN-graph reinforcemente learning environment we created for neural network pruning. 
Red arrows show that the process starts from the original DNN. The model size evaluator first evaluates the size of the DNN. If the size is not satisfied, the graph generator converts the DNN into a hierarchical computational graph. Then, the GNN-RL agent leverages m-GNN to learn pruning ratios from the graph. The pruner prunes the DNN with the pruning ratios and begins the next iteration from the compressed DNN. 
Each step of the compression will change the network topology. Thus, the DNN-Graph environment reconstructs a new hierarchical computational graph for the GNN-RL agent corresponding to the current compression state.
Once the compressed DNN satisfies the size constraint, the evaluator will end the episode, and the accuracy evaluator will assess the pruned DNN's accuracy as an episode reward for the GNN-RL agent.

GNN-RL defualt policy network ```gnnrl.graph_env.graph_environment```.

    from gnnrl.graph_env.graph_environment import graph_env
    env = graph_env(net,n_layer,args.dataset,val_loader,args.compression_ratio,args.g_in_size,args.log_dir,input_x,device,args)

## Fine-tuning