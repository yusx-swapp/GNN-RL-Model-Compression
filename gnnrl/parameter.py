import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='gnnrl search script')

    # datasets and model
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
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
    parser.add_argument('--max_episodes', default=15000, type=int, help='max training episodes')
    parser.add_argument('--max_timesteps', default=5, type=int, help='max timesteps in one episode')
    parser.add_argument('--action_std', default=0.5, type=float, help='constant std for action distribution (Multivariate Normal)')
    parser.add_argument('--K_epochs', default=10, type=int, help='update policy for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float, help='clip parameter for RL')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate for optimizer')
    parser.add_argument('--update_timestep', default=100, type=int, help='update policy every n timesteps')

    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=4, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')


    parser.add_argument('--log_dir', default='./logs', type=str, help='log dir')
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--transfer', action='store_true', help='topology transfer')
    # parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    # parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    # parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')

    return parser.parse_args()