# Introduction By Example

## Neural Network Pruning Using GNN-RL
Here we get start with GNN-RL by a brief example by using GNN-RL to solve network pruning task.
Below are steps:

1. load target neural network    
2. define GNN-RL environment
3. define GNN-RL reinforcement learning agent
4. search an training the RL agent for pruning policy


## Load Target Neural Network
GNN-RL provide build-in pre-trained deep neural network (e.g ResNet-20/32-56), can be easily load by ``load_model`` .

    device = torch.device(args.device)
    net = load_model(args.model,args.data_root)
    net.to(device)

Iterate the network and get the number of convolutional layers (convolutional layers are ready to be pruned by GNN-RL),
    
    n_layer = 0
    for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer +=1

Or you can use GNN-RL build-in function ``get_num_hidden_layer``,  which can easily get the number of target pruning layer of DNNs,

    n_layer = get_num_hidden_layer(net,args.model)

Then, load the DNN's corresponding validation dataset (used to caculate rewards),

    train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
                                                        data_root=args.data_root)

## DNN-Graph Environment for Neural Network Pruning
GNN-RL modeling the DNN as a computatuinal graph, and constructs a graph environment to simulation the topology change when pruning. 

    env = graph_env(net,n_layer,args.dataset,val_loader,args.compression_ratio, 
                                args.g_in_size,args.log_dir,input_x,device,args)

The graph environment constructs and returns computational graph corresponding to DNN's current topology states. Meanwhile, the graph environment evaluates the size of DNN (e.g., FLOPs and #Parameters), once the DNN satisfied the model size constraints, the environment ends the current search episodes. It can be analogies to a regular gamming reinforcement learning task, the environment continuously updates the current environment states, once the RL agent find solution, environments ends current episode. 
<!-- *Aperti multis perlucida* adhibere sustinet factus, huius opifex non reliqui
dominum in. Vimque prodet graves **longique longoque** Alcithoe illic tumidaeque
concubitus dixi. Nox dentes iram quacumque parte crurumque patrem, at formosus
limite. Sine ipse [bovem](http://www.habenasdixit.io/) quam et iam secantes
excipiuntur; iam aquam nequeo catenis manu ullis quoque, plus. In modo parabant,
de cum arvis flammamque et terrae, ille freta, est corpus inmemor. -->
## Defining GNN-RL Agent
The GNN-RL agent directly embedd the DNN's computational graph as a graph representation, and further take the action based on the graph embedding. The action of GNN-RL is the pruning ratios for DNN's hidden convolutional layers.

    agent = Agent(state_dim=args.g_in_size, action_dim=layer_share, action_std = args.action_std, lr = args.lr, betas = args.beta,
                     gamma = args.gamma, K_epochs = args.K_epochs, eps_clip = args.eps_clip)

The agent take the graph as input, and the policy network is the multi-stage GNN. The agent updates through DDPG reinforcement learning process.

## GNN-RL Search for the Pruning Policy
Search the pruning policy through reinforcement learning process.

    ############## Hyperparameters ##############
    env_name = "gnnrl_search"
    render = False
    solved_reward = solved_reward  # stop training if avg_reward > solved_reward
    log_interval = log_interval  # print avg reward in the interval
    max_episodes = max_episodes  # max training episodes
    max_timesteps = max_timesteps  # max timesteps in one episode

    update_timestep = update_timestep  # update policy every n timesteps
    # lr = lr  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = random_seed
    #############################################


    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()


    # print("Learning rate: ", lr, '\t Betas: ', betas)

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

GNN-RL provides easy way to search and train the policy network, by calling the ```search```

    search(env,agent, update_timestep=args.update_timestep,max_timesteps=args.max_timesteps, max_episodes=args.max_episodes,
           log_interval=10, solved_reward=args.solved_reward, random_seed=args.seed)
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
