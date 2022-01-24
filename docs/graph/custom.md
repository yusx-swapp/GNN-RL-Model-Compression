
## Create your customized graph construction method

In this subsection we give the example for model VGG-16 as computational graph use Torch-Geometric package. Create the graph by convolutional layer's in and out channels. Write a class for converting DNN to computational graph:
```python
from gnnrl.graph_env.graph_construction import conv_motif,conv_sub_graph
from gnnrl.utils.batchwise_graphs import get_next_graph_batch
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

class computational_graph_pyg():
    def __init__(self, in_channels, out_channels, feature_size, device=None, pruning_ratios=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pruning_ratios = pruning_ratios
        self.c_in_channels = self.in_channels[1:] * pruning_ratios
        self.c_out_channels = self.out_channels[:-1] * pruning_ratios
        self.feature_size = feature_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.plain_graph = self.plain_computational_graph()
        self.hierachical_graph = self.hierarchical_computational_graph()

    def plain_computational_graph(self):
        return None

    def hierarchical_computational_graph(self):

        hierarchical_graph={}
        hierarchical_graph['level1'] = self.level1_graph().to(self.device)
        hierarchical_graph['level2'] = self.level2_graph().to(self.device)
        return hierarchical_graph

    def level1_graph(self):

        level_1_graphs = []

        for in_c in self.c_in_channels:
            edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,self.feature_size]).to(self.device)
            level_1_graphs.append(G.to(self.device))

        level_1_graphs = DataLoader(level_1_graphs,batch_size=len(level_1_graphs), shuffle=False)
        level_1_graphs = get_next_graph_batch(level_1_graphs)


        return level_1_graphs

    def level2_graph(self):

        node_cur = 0
        edge_list = []
        edge_type = []

        k = 0   # layer index

        normal_ope_edge_type = len(self.c_out_channels)
        for i in range(len(self.c_out_channels)):

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,self.c_out_channels[i],edge_list,edge_type,i,normal_ope_edge_type)
            k+=1
            #Batch Norm
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(normal_ope_edge_type)
            node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)


        Graph.x = torch.randn([Graph.num_nodes, self.feature_size])
        Graph.edge_features = None
        Graph = DataLoader([Graph],batch_size=1, shuffle=False)
        Graph = get_next_graph_batch(Graph)

        return Graph

    def update_pruning_ratio(self,pruning_ratios):
        self.pruning_ratios = pruning_ratios
        self.c_in_channels = self.in_channels[1:] * pruning_ratios
        self.c_out_channels = self.out_channels[:-1] * pruning_ratios
        self.plain_graph = self.plain_computational_graph()
        self.hierachical_graph = self.hierarchical_computational_graph()
```

Then load the neural network from `torchvision.models`,
```python
from torchvision.models import vgg16
net = vgg16()
```
Next, get the network's convolutional layers' input and output channels,
```python
import torch.nn as nn
in_channels = []
out_channels=[]
for name,layer in net.named_modules():
    if isinstance(layer,nn.Conv2d):
        in_channels.append(layer.in_channels)
        out_channels.append(layer.out_channels)
```
Initialize the ```computational_graph_pyg``` object you created,
```python
pyg_g = computational_graph_pyg(in_channels,out_channels,feature_size=20)
```
Get your computational graph by calling the function ```plain_computational_graph()```,
```python
graph = pyg_g.plain_computational_graph()
```





[comment]: <> (    class SyntheticDataset&#40;DGLDataset&#41;:)

[comment]: <> (            def __init__&#40;self&#41;:)

[comment]: <> (                super&#40;&#41;.__init__&#40;name='synthetic'&#41;)

[comment]: <> (            def process&#40;self&#41;:)

[comment]: <> (                edges = pd.read_csv&#40;'./graph_edges.csv'&#41;)

[comment]: <> (                properties = pd.read_csv&#40;'./graph_properties.csv'&#41;)

[comment]: <> (                self.graphs = [])

[comment]: <> (                self.labels = [])

[comment]: <> (                # Create a graph for each graph ID from the edges table.)

[comment]: <> (                # First process the properties table into two dictionaries with graph IDs as keys.)

[comment]: <> (                # The label and number of nodes are values.)

[comment]: <> (                label_dict = {})

[comment]: <> (                num_nodes_dict = {})

[comment]: <> (                for _, row in properties.iterrows&#40;&#41;:)

[comment]: <> (                    label_dict[row['graph_id']] = row['label'])

[comment]: <> (                    num_nodes_dict[row['graph_id']] = row['num_nodes'])

[comment]: <> (                # For the edges, first group the table by graph IDs.)

[comment]: <> (                edges_group = edges.groupby&#40;'graph_id'&#41;)

[comment]: <> (                # For each graph ID...)

[comment]: <> (                for graph_id in edges_group.groups:)

[comment]: <> (                    # Find the edges as well as the number of nodes and its label.)

[comment]: <> (                    edges_of_id = edges_group.get_group&#40;graph_id&#41;)

[comment]: <> (                    src = edges_of_id['src'].to_numpy&#40;&#41;)

[comment]: <> (                    dst = edges_of_id['dst'].to_numpy&#40;&#41;)

[comment]: <> (                    num_nodes = num_nodes_dict[graph_id])

[comment]: <> (                    label = label_dict[graph_id])

[comment]: <> (                    # Create a graph and add it to the list of graphs and labels.)

[comment]: <> (                    g = dgl.graph&#40;&#40;src, dst&#41;, num_nodes=num_nodes&#41;)

[comment]: <> (                    self.graphs.append&#40;g&#41;)

[comment]: <> (                    self.labels.append&#40;label&#41;)

[comment]: <> (                # Convert the label list to tensor for saving.)

[comment]: <> (                self.labels = torch.LongTensor&#40;self.labels&#41;)

[comment]: <> (            def __getitem__&#40;self, i&#41;:)

[comment]: <> (                return self.graphs[i], self.labels[i])

[comment]: <> (            def __len__&#40;self&#41;:)

[comment]: <> (                return len&#40;self.graphs&#41;)
