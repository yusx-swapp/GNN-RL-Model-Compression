# import urllib

import dgl
import pandas as pd
import torch
import torch.nn as nn
from dgl.data import DGLDataset
from torch_geometric.data import Data
import numpy as np
import logging
from torch_geometric.data import DataLoader

from utils.batchwise_graphs import get_next_graph_batch

logging.disable(30)

def net_info(model_name):
    if model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        n_layer = 3
        n_blocks = 9
        in_channels = [3]
        out_channels = [16]
        channels=[[16],[32],[64]]
        if model_name == "resnet56":
            n_layer = 3
            n_blocks = 9
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]
        elif model_name == "resnet44":
            n_layer = 3
            n_blocks = 7
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]
        elif model_name == "resnet110":
            n_layer = 3
            n_blocks = 18
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        elif model_name == "resnet32":
            n_layer = 3
            n_blocks = 5
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        elif model_name == "resnet20":
            n_layer = 3
            n_blocks = 3
            in_channels = [3]
            out_channels = [16]
            channels=[[16],[32],[64]]

        for i in range(n_layer):
            if i == 0:
                in_channels.extend(channels[i]*(n_blocks*2+1))
            elif i==2:
                in_channels.extend(channels[i]*(n_blocks*2-1))
            else:
                in_channels.extend(channels[i]*n_blocks*2)

            out_channels.extend(channels[i]*n_blocks*2)
        return in_channels,out_channels,n_blocks
    elif model_name == 'mobilenetv2':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from torchvision.models import mobilenet_v2
        net = mobilenet_v2()
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)


        return in_channels,out_channels,depth_wise
    elif model_name == 'mobilenet':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)


        return in_channels,out_channels,depth_wise
    elif model_name == 'shufflenet':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)
        return in_channels,out_channels,depth_wise
    elif model_name == 'shufflenetv2':
        in_channels = []
        out_channels=[]
        depth_wise=[]

        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
                if layer.groups == layer.in_channels:
                    depth_wise.append(True)
                else:
                    depth_wise.append(False)
        return in_channels,out_channels,depth_wise
    elif model_name == 'vgg16':
        in_channels = []
        out_channels=[]

        from torchvision.models import vgg16
        net = vgg16()
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
        return in_channels,out_channels,[]
    elif model_name == 'resnet18':
        in_channels = []
        out_channels=[]

        from torchvision.models import resnet18
        net = resnet18()
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
        return in_channels,out_channels,[]

    elif model_name == 'resnet50':
        in_channels = []
        out_channels=[]

        from torchvision.models import resnet50
        net = resnet50()
        for name,layer in net.named_modules():
            if isinstance(layer,nn.Conv2d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
        return in_channels,out_channels,[]
def create_edge_features(edge_types,type_features,device):
    if max(edge_types)> len(type_features):
        #random initial primitive operation like batch norm
        type_features = torch.cat((type_features,torch.randn((max(edge_types)+1 - len(type_features),type_features.shape[1]),requires_grad=True).to(device)),dim=0)

    for i in range(len(edge_types)):
        if i == 0:
            edge_features = type_features[edge_types[i]].unsqueeze(0)
        else:
            edge_features = torch.cat((edge_features,type_features[edge_types[i]].unsqueeze(0)),dim=0)
    return edge_features
def conv_sub_graph(node_cur,n_filter,edge_list,edge_type,conv_type=None,concat_type=None):
    '''
    Construct a subgraph for conv operation in a DNN
    :param node_cur:
    :param n_filter:
    :param edge_list:
    :return:
    '''
    #node_features.append(torch.randn(feature.size()))
    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i+1)])
        edge_type.append(conv_type)

        edge_list.append([node_cur + (i+1), node_cur + n_filter +1])
        edge_type.append(concat_type)

    node_cur += n_filter + 1

    return edge_list,edge_type,node_cur
def depth_sub_graph(n_filter,node_cur=0):
    edge_list = []
    for i in range(n_filter):
        edge_list.append([node_cur, node_cur + (i+1)])
        edge_list.append([node_cur + (i+1), node_cur + n_filter + (i+1)])
        edge_list.append([node_cur + n_filter+ (i+1), node_cur + 2*n_filter + 1])

    node_cur += 2 * n_filter + 1

    return np.array(edge_list),node_cur
def conv_motif(n_in_chanel,node_cur = 0):
        '''
        Construct a subgraph for conv operation with n input channels in a DNN
        :param node_cur:
        :param n_filter:
        :param edge_list:
        :return:
        '''
        edge_list = []

        #node_features.append(torch.randn(feature.size()))
        if n_in_chanel == 0:
            n_in_chanel = 1
        for i in range(n_in_chanel):
            edge_list.append([node_cur, node_cur + (i+1)])
            # edge_type.append(type_dict['split'])
            edge_list.append([node_cur + (i+1), node_cur + n_in_chanel +(i+1)])
            edge_list.append([node_cur + n_in_chanel +(i+1),node_cur + 2*n_in_chanel +1])

        return np.array(edge_list)

def level1_graph(in_channel,feature_size,net_name='resnet110',device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    level_1_graphs = []
    if net_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        for in_c in in_channel:
            # conv 3*3 with 3 in channel
            edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,feature_size]).to(device)
            level_1_graphs.append(G.to(device))

    elif net_name in ['resnet18','resnet50']:
        for in_c in in_channel:
            # conv 3*3 with 3 in channel
            edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,feature_size]).to(device)
            level_1_graphs.append(G.to(device))

    elif net_name == 'mobilenet':
        _, __ ,depth_wise = net_info(net_name)
        for i,in_c in enumerate(in_channel):
            if depth_wise[i]:
                edge_index,_ = depth_sub_graph(in_c)
            else:
                edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,feature_size]).to(device)
            level_1_graphs.append(G.to(device))

    elif net_name == 'mobilenetv2':
        _, __ ,depth_wise = net_info(net_name)
        for i,in_c in enumerate(in_channel):
            if depth_wise[i]:
                edge_index,_ = depth_sub_graph(in_c)
            else:
                edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,feature_size]).to(device)
            level_1_graphs.append(G.to(device))
    elif net_name == 'vgg16':
        for in_c in in_channel:
            edge_index = conv_motif(in_c)

            G = Data(edge_index=torch.tensor(edge_index).long().t().contiguous())
            G.x = torch.randn([G.num_nodes,feature_size]).to(device)
            level_1_graphs.append(G.to(device))
    level_1_graphs = DataLoader(level_1_graphs,batch_size=len(level_1_graphs), shuffle=False)
    level_1_graphs = get_next_graph_batch(level_1_graphs)


    return level_1_graphs
def level2_graph(type_dict,out_channels,net_name,n_features=20,device=None):

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_cur = 0
    edge_list = []
    edge_type = []

    k = 0   # layer index
    if net_name == 'vgg16':
        in_channels,out_channels,_ = net_info(net_name)

        for i in range(len(out_channels)):

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[i],edge_list,edge_type,i,type_dict['concatenates'])
            k+=1
            #Batch Norm
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['ReLu'])
            node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)

    elif net_name == 'mobilenet':
        _, __ ,depth_wise = net_info(net_name)
        for i in range(len(out_channels)):
            if depth_wise[i]:
                # edge_list,edge_type,node_cur = depth_sub_graph(node_cur,out_channels[k],edge_list,edge_type,type_dict['conv3x3x3'],type_dict['concatenates'])
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(i)

                node_cur += 1

                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[i],edge_list,edge_type,i,type_dict['concatenates'])

                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
    elif net_name == 'mobilenetv2':
        _, __ ,depth_wise = net_info(net_name)
        for i in range(len(out_channels)):
            if depth_wise[i]:
                edge_list.append([node_cur,node_cur+1])
                edge_type.append(i)

                node_cur += 1

                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[i],edge_list,edge_type,i,type_dict['concatenates'])

                edge_list.append([node_cur,node_cur+1])
                edge_type.append(type_dict['ReLu'])
                node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
    elif net_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        # in_channels,out_channels,blocks = net_info(net_name)
        _,_,blocks = net_info(net_name)


        ## conv 16*3*3 chanel 64
        edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
        k+=1
        #Batch Norm
        edge_list.append([node_cur,node_cur+1])
        edge_type.append(type_dict['bacthNorm'])
        node_cur += 1

        node_cur_temp = node_cur

        for i in range(blocks):
            #basic resnet blocks
            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur

        for i in range(blocks):
            #basic resnet blocks
            if i == 0:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            if i == 0:
                edge_type.append(type_dict['shortCut2'])
            else:
                edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur


        for i in range(blocks):
            #basic resnet blocks
            if i == 0:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            else:
                edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[k],edge_list,edge_type,k,type_dict['concatenates'])
            k+=1
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['bacthNorm'])
            node_cur += 1

            #shrot cuts
            edge_list.append([node_cur_temp,node_cur])
            if i == 0:
                edge_type.append(type_dict['shortCut2'])
            else:
                edge_type.append(type_dict['shortCut1'])
            node_cur_temp = node_cur

        #Linear
        edge_list.append([node_cur,node_cur+1])
        edge_type.append(type_dict['linear'])
        # Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)
    elif net_name in ['resnet18','resnet50']:
        in_channels,out_channels,_ = net_info(net_name)

        for i in range(len(out_channels)):

            edge_list,edge_type,node_cur = conv_sub_graph(node_cur,out_channels[i],edge_list,edge_type,i,type_dict['concatenates'])
            k+=1
            #Batch Norm
            edge_list.append([node_cur,node_cur+1])
            edge_type.append(type_dict['ReLu'])
            node_cur += 1
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous(),edge_type =edge_type)



    Graph.x = torch.randn([Graph.num_nodes, n_features])
    Graph.edge_features = None
    Graph = DataLoader([Graph],batch_size=1, shuffle=False)
    Graph = get_next_graph_batch(Graph)

    return Graph

def hierarchical_graph_construction(in_channels,out_channels,net_name,n_features=20,device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    level2_type_dict = {
        #the conv in each layer have dif types, starts from 0
        "concatenates":len(out_channels),
        "shortCut1":len(out_channels)+1,
        "shortCut2":len(out_channels)+2,
        "bacthNorm":len(out_channels)+3,
        "linear":len(out_channels)+4,
        "ReLu":len(out_channels)+5
    }
    hierarchical_graph={}
    hierarchical_graph['level1'] = level1_graph(in_channels,n_features,net_name,device).to(device)
    hierarchical_graph['level2'] = level2_graph(level2_type_dict,out_channels,net_name,n_features,device).to(device)
    return hierarchical_graph


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
        Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous())


        Graph.x = torch.randn([Graph.num_nodes, self.feature_size])


        return Graph

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


class computational_graph_dgl():
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
        # self.hierachical_graph = self.hierarchical_computational_graph()

    def plain_computational_graph(self):
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

        edge_list = np.array(edge_list)
        edge_list = edge_list.transpose(1,0)
        source = edge_list[0]
        dist = edge_list[1]
        Graph = dgl.graph((source, dist))
        # Graph = Data(edge_index=torch.tensor(edge_list).t().contiguous())
        #
        #
        # Graph.x = torch.randn([Graph.num_nodes, self.feature_size])
        #
        #
        return Graph

class motif(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        edges = pd.read_csv('./graph_edges.csv')
        properties = pd.read_csv('./graph_properties.csv')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)



# Thumbnail credits: (Un)common Use Cases for Graph Databases, Michal Bachman
# sphinx_gallery_thumbnail_path = '_static/blitz_6_load_data.png'

if __name__ == '__main__':
    # urllib.request.urlretrieve(
    #     'https://data.dgl.ai/tutorial/dataset/graph_edges.csv', './graph_edges.csv')
    # urllib.request.urlretrieve(
    #     'https://data.dgl.ai/tutorial/dataset/graph_properties.csv', './graph_properties.csv')
    # edges = pd.read_csv('./graph_edges.csv')
    # properties = pd.read_csv('./graph_properties.csv')
    #
    # edges.head()
    #
    # properties.head()
    #
    # dataset = motif()
    # graph, label = dataset[0]
    # print(graph, label)

    in_channels = [3,16,32]
    out_channels = [16,32,32]
    feature_size = 20
    dgl_g = computational_graph_dgl(in_channels,out_channels,feature_size)
