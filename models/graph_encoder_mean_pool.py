import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.multi_stage_gcn import multi_stage_conv as gcn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class stage2_gcn_encoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(stage2_gcn_encoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature


        self.conv1 = gcn(in_feature, hidden_feature)

        self.linear1 = nn.Linear(hidden_feature,out_feature)

        self.tanh = F.tanh
        self.dropout = F.dropout

        self.pool = global_mean_pool

    def forward(self,Graph):
        x, edge_index,edge_features,batch = Graph.x, Graph.edge_index,Graph.edge_features,Graph.batch



        x = self.conv1(x,edge_index,edge_features)
        x = self.tanh(x)
        x = self.dropout(x, training=self.training)
        node_embeddings = x
        #x = x.mean(dim=0).reshape(1,-1)
        graph_embeddings = global_mean_pool(x,batch)

        graph_embeddings = self.linear1(graph_embeddings)
        graph_embeddings = self.tanh(graph_embeddings)

        return graph_embeddings,node_embeddings

class stage1_gcn_encoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(stage1_gcn_encoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature


        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature,out_feature)
        self.tanh = F.tanh
        self.dropout = F.dropout

        self.pool = global_mean_pool

    def forward(self,Graph):
        x, edge_index,batch = Graph.x, Graph.edge_index,Graph.batch



        x = self.conv1(x,edge_index)
        x = self.tanh(x)
        # x = self.dropout(x, training=self.training)

        node_embeddings = x

        graph_embeddings = global_mean_pool(x,batch)

        graph_embeddings = self.linear1(graph_embeddings)
        graph_embeddings = self.tanh(graph_embeddings)

        return graph_embeddings,node_embeddings

class multi_stage_graph_encoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(multi_stage_graph_encoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        # self.device = device

        self.stage1_encoder = stage1_gcn_encoder(in_feature,hidden_feature, in_feature)
        self.stage2_encoder = stage2_gcn_encoder(in_feature,hidden_feature, out_feature)



    def forward(self,  hierarchical_graphs):
        #two stage embedding
        if type(hierarchical_graphs) is not list:
            hierarchical_graphs = [hierarchical_graphs]


        for i,hierarchical_graph in enumerate(hierarchical_graphs):
            level_1_graphs = hierarchical_graph['level1']
            level_2_graph = hierarchical_graph['level2']
            graph_embeddings,node_embeddings = self.stage1_encoder(level_1_graphs)

            from graph_env.graph_construction import create_edge_features
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            level_2_graph.edge_features = create_edge_features(level_2_graph.edge_type[0],graph_embeddings,device)

            hierarchical_embedding, node_embeddings = self.stage2_encoder(level_2_graph)
            if i == 0:
                hierarchical_embeddings = hierarchical_embedding
            else:
                torch.cat((hierarchical_embeddings,hierarchical_embedding),dim=0)

        return hierarchical_embeddings



