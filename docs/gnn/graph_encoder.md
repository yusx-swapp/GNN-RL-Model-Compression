# GCN-based graph encoder
GNN-RL uses GCN-based policy network directly learn topology from graphs.
GNN-RL supports Torch-Geometric and DGL graph nueral network backend. 
## Embedding computational graph use DGL
First, create a neural network and get the network information.

    net = load_model(args.model,args.data_root)
    net.to(device)
    in_channels, out_channels = graph_construction.net_info(net)

Then, construct corresponding computational graph by create an ```computational_graph_dgl``` object,and create a plain simplified computational graph, call ```plain_computational_graph(self)```,

    dgl_g = computational_graph_dgl(in_channels,out_channels,feature_size)
    graph = dgl_g.plain_computational_graph()

Embedding the computational graph,

    mode = graph_encoder_dgl(feature_size,hidden_feature,out_feature)
    embedding = mode(graph)




## Embedding computational graph use pyg
Construct corresponding computational graph by create an ```computational_graph_pyg``` object,and create a plain simplified computational graph, call ```plain_computational_graph(self)```,

    pyg_g = computational_graph_pyg(in_channels,out_channels,feature_size)
    graph = pyg_g.plain_computational_graph()

Embedding the computational graph,

    mode = graph_encoder_pyg(feature_size,hidden_feature,out_feature)
    embedding = mode(graph)

## Build your own GCN encoder

Define the graph encoder using pyg:

    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool

    class graph_encoder_pyg(nn.Module):
        def __init__(self,in_feature, hidden_feature, out_feature):
            super(graph_encoder_pyg, self).__init__()

            self.in_feature = in_feature
            self.hidden_feature = hidden_feature
            self.out_feature = out_feature


            self.conv1 = GCNConv(in_feature, hidden_feature)
            self.linear1 = nn.Linear(hidden_feature,out_feature)
            self.tanh = nn.Tanh()
            self.relu = torch.relu

        def forward(self,  Graph):

            x, edge_index,batch = Graph.x, Graph.edge_index,Graph.batch
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            embedding = global_mean_pool(x,batch)
            embedding = self.linear1(embedding)
            embedding = self.tanh(embedding)

            return embedding

