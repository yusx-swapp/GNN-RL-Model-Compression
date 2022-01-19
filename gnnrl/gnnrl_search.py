import torch.backends.cudnn as cudnn
import torch
import logging
import numpy as np
from lib.RL.agent import Memory
logging.disable(30)

torch.backends.cudnn.deterministic = True
def transfer_policy_search():
    return

def search(env,agent, update_timestep,max_timesteps, max_episodes,
           log_interval=10, solved_reward=None, random_seed=None):

    ############## Hyperparameters ##############
    env_name = "gnnrl_search"
    render = False
    solved_reward = solved_reward  # stop training if avg_reward > solved_reward
    log_interval = log_interval  # print avg reward in the interval
    max_episodes = max_episodes  # max training episodes
    max_timesteps = max_timesteps  # max timesteps in one episode

    update_timestep = update_timestep  # update policy every n timesteps

    random_seed = random_seed
    #############################################


    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()



    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

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

        # stop training if avg_reward > solved_reward
        if (i_episode % log_interval) != 0 and running_reward / (i_episode % log_interval) > (solved_reward):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './rl_solved_{}.pth'.format(env_name))
            break

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

