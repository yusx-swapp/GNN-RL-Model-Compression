# Multi-stage graph neural network
DNN's computational graph contains multiple motif (reused graph pattern), GNN-RL can model DNN as a hierarchical computational graph and uses efficient multi-stage graph embedding.

Define the neural network, and get its information. 
    
    net = load_model(args.model,args.data_root)
    net.to(device)
    in_channels, out_channels = graph_construction.net_info(net)

## Multi-stage graph embedding

### DGL backend

Create a hierarchical computational graph use DGL backend, call ```hierarchical_computational_graph(self)```,

    graph = dgl_g.hierarchical_computational_graph()

Define the encoder by creating the ```multi_stage_graph_encoder_dgl``` object,

    encoder = multi_stage_graph_encoder_dgl(in_feature, hidden_feature, out_feature)
    hierarchical_embeddings = encoder(graph)


### Torch-Geometric backend

Create a hierarchical computational graph use DGL backend, call ```hierarchical_computational_graph(self)```,

    graph = pyg_g.hierarchical_computational_graph()

Define the encoder by creating the ```multi_stage_graph_encoder_pyg``` object,

    encoder = multi_stage_graph_encoder_pyg(in_feature, hidden_feature, out_feature)
    hierarchical_embeddings = encoder(graph)