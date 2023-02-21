# GNN-RL pipeline

## Overview

GNN-RL pipleline is a toolkit to help users extract topology information through reinforcement learning task (e.g., topology-aware neural network compression and job scheduling).
GNN-RL contains two major part -- graph neural network (GNN) and reinforcement learning (RL). It requires your research object is a graph or been modeled as a graph, the reinforcement learning take the graph as environment states and produces actions by using a GNN-based policy network.
We have successfully tested the GNN-RL on model compression task, where we model a deep neural network (DNN) as a graph, and search compression policy using reinforcement learning.


## GNN-RL on Model Compression

We have tested our GNN-RL on fine-grained pruning and structured pruning on CNNs.
We first model the trage DNN as a computational graph, and GNN-RL to search pruning policy directly through DNN's topology. In the reinforcement learning task definition, we use the computational graph as environment states, the action spaces are defined as pruning ratios, and the rewards are the compressed DNN's accuracy. Once the compressed model size satisfied the resource requirements, GNN-RL end the search episode. The network pruning task can be visualize as the diagram below.


## Apply GNN-RL Pipeline on Other Discipline
The GNN-RL is not limited to model compression, it aims to extract topology information through reinforcement learning task.
We provide APIs to help you define your customized job!  It only requires your research object is a graph or been modeled as a graph, you can define your customized RL task (e.g., action space, environment states, rewards). Currently, our collegues are testing GNN-RL on job scheduling taks.


## Page Index
Here is the index for you to quick locate the GNN-RL:

1. [Installation](installation.md)
2. [Introduction by example](intro.md)
3. [Model DNN as graph](graph/example.md)
4. [GNN-RL in model compression](compression/pruning.md)


## Related Papers
If you find our useful or use our work in your job, please cite the paper:



    @InProceedings{yu2022gnnrl,
      title     = {Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning},
      author    = {Yu, Sixing and Mazaheri, Arya and Jannesari, Ali},
      booktitle = {Proceedings of the 39th International Conference on Machine Learning},
      pages     = {25656--25667},
      year      = {2022},
      volume    = {162},
      series    = {Proceedings of Machine Learning Research},
      month     = {17--23 Jul},
      publisher = {PMLR},
    }


    
    @InProceedings{yu2021agmc,
        author    = {Yu, Sixing and Mazaheri, Arya and Jannesari, Ali},
        title     = {Auto Graph Encoder-Decoder for Neural Network Pruning},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {6362-6372}
    }

    @article{yu2021spatl,
      title={SPATL: Salient Parameter Aggregation and Transfer Learning for Heterogeneous Clients in Federated Learning},
      author={Yu, Sixing and Nguyen, Phuong and Abebe, Waqwoya and Anwar, Ali and Jannesari, Ali},
      journal={arXiv preprint arXiv:2111.14345},
      year={2021}
    }