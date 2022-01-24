
## Supported packages
GNN-RL support popular deep graph neural network package, such as [Torch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [DGL](https://www.dgl.ai/).
In this section we will give examples to modeling DNN's topology to computational graph, and embedding them using graph neural network.

## Create hierarchical computational graph using Torch-geometric
GNN-RL also support Torch-Geometric packadge, and also provid easy-to-use function to create Torch-Geometric graph object.
First, get the information of target DNN. The build-in function ```graph_construction.net_info(net_name)``` can automatically process the DNN and return the input and output channels of DNN for constructing a graph.
```python
from gnnrl.utils.load_networks import load_model
from gnnrl.graph_env import graph_construction
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet20', '.')
net = net.to(device)
in_channels, out_channels, _ = graph_construction.net_info('resnet20')
```

Or you can write your own function to get the network's convolutional layers' input and output channels,
```python
import torch.nn as nn
in_channels = []
out_channels=[]
for name,layer in net.named_modules():
    if isinstance(layer,nn.Conv2d):
        in_channels.append(layer.in_channels)
        out_channels.append(layer.out_channels)
```

Then, construct the graph by create an ```computational_graph_pyg``` object,
```python
from gnnrl.graph_env.graph_construction import computational_graph_pyg
pyg_g = computational_graph_pyg(in_channels,out_channels,feature_size=10)
```

### Hierarchical computational graph
Import DGL backend method and convert DNN to simplified computational graph.
To create a plain simplified computational graph, call ```hierarchical_computational_graph()```,
```python
graph = pyg_g.hierarchical_computational_graph()
```


## Create computational graph using DGL
First, get the information of target DNN. The build-in function ```graph_construction.net_info(net)``` can automatically process the DNN and return the input and output channels of DNN for constructing a graph.
```python
from gnnrl.utils.load_networks import load_model
from gnnrl.graph_env import graph_construction
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('resnet20', '.')
net = net.to(device)
in_channels, out_channels, _ = graph_construction.net_info('resnet20')
```


Then, construct the graph by create an ```computational_graph_dgl``` object,
```python
from gnnrl.graph_env.graph_construction import computational_graph_dgl
dgl_g = computational_graph_dgl(in_channels,out_channels,feature_size=10)
```

### Hierarchical computational graph
Import DGL backend method and convert DNN to simplified computational graph.
To create a plain simplified computational graph, call ```plain_computational_graph(self)```,
```python
graph = dgl_g.hierarchical_computational_graph()
```


