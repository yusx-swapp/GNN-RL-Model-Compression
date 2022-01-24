
from gnnrl.utils.load_networks import load_model
from gnnrl.graph_env import graph_construction
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet20', '.')
net = net.to(device)
in_channels, out_channels, _ = graph_construction.net_info('resnet20')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet20', '.')
net.to(device)
in_channels, out_channels = graph_construction.net_info(net)

from gnnrl.graph_env.graph_construction import computational_graph_dgl
dgl_g = computational_graph_dgl(in_channels,out_channels,feature_size=10)
in_channels = [3,16,32]
out_channels = [16,32,32]
feature_size = 20
dgl_g = computational_graph_dgl(in_channels,out_channels,feature_size)