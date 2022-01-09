
## Create your customized graph construction method

In this subsection we give the example for model VGG-16 as computational graph use Torch-Geometric package. Create the graph by convolutional layer's in and out channels. Write a class for converting DNN to computational graph:

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


Then define an object and construct a computational graph for VGG-16:

    in_channels = [3,16,32]
    out_channels = [16,32,32]
    feature_size = 20
    computational_graph_pyg(in_channels,out_channels,feature_size)






    class SyntheticDataset(DGLDataset):
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
