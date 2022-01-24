import os

import torch as T,torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from gnnrl.models.graph_encoder import multi_stage_graph_encoder
from gnnrl.models.graph_encoder_plain import graph_encoder_pyg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
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

class ActorNetworkPlain(nn.Module):
    def __init__(self,g_in_size, g_hidden_size, g_embedding_size,hidden_size, nb_actions,
                 chkpt_dir='tmp/rl'):
        super(ActorNetworkPlain, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_rl')
        self.graph_encoder = graph_encoder_pyg(g_in_size, g_hidden_size, g_embedding_size)
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

class CriticNetworkPlain(nn.Module):
    def __init__(self, g_in_size, g_hidden_size, g_embedding_size,
                 chkpt_dir='tmp/rl'):
        super(CriticNetworkPlain, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_rl')
        self.graph_encoder_critic = graph_encoder_pyg(g_in_size, g_hidden_size, g_embedding_size)
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



class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_states = memory.states
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()


        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            if _ % 5 == 0:
                print('Epoches {} \t loss: {} \t '.format(_, loss.mean()))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        

            

    
