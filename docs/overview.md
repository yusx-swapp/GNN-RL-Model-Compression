## GNN-RL pipeline

GNN-RL pipleline is a toolkit to help users extract topology information through reinforcement learning task (e.g., topology-aware neural network compression and job scheduling).
GNN-RL contains two major part -- graph neural network (GNN) and reinforcement learning (RL). It requires your research object is a graph or been modeled as a graph, the reinforcement learning take the graph as environment states and produces actions by using a GNN-based policy network.
We have successfully tested the GNN-RL on model compression task, where we model a deep neural network (DNN) as a graph, and search compression policy using reinforcement learning.


## GNN-RL on Model Compression

We have tested our GNN-RL on fine-grained pruning and structured pruning on CNNs. 
We first model the trage DNN as a computational graph, and GNN-RL to search pruning policy directly through DNN's topology. In the reinforcement learning task definition, we use the computational graph as environment states, the action spaces are defined as pruning ratios, and the rewards are the compressed DNN's accuracy. Once the compressed model size satisfied the resource requirements, GNN-RL end the search episode. The network pruning task can be visualize as the diagram below.
<!-- Se validi [marmor](http://www.subibis.net/) non si quoque minuuntur tergo,
revelli **tenebris**, apex *Tethys* rogarem temperius monte altaque cura. Gratia
molliri tempore, tanto mugitibus ictus. Iunctum *requirere probat* destinat
vigore?

- Mea per dum ruent invita quos et
- Dicentum nece
- Sibi iuro omnia sentit in timeo brevissimus
- Misit adflat suum inposito vocem illic figuris -->

## Apply GNN-RL Pipeline on Other Discipline
The GNN-RL is not limited to model compression, it aims to extract topology information through reinforcement learning task.
We provide APIs to help you define your customized job!  It only requires your research object is a graph or been modeled as a graph, you can define your customized RL task (e.g., action space, environment states, rewards). Currently, our collegues are testing GNN-RL on job scheduling taks.
<!-- 
Rogos indotata geminas gaudebat ferendo, nemus quod multum lumina invocat
tempora nebulae, et agnoscis! Pudori vulnere. Celerem festinus: delere currum
venerabile limina spatiantia vastum, concita, mei Aeacides, et dea nefas. Artis
fuit ille nostri quater lumina nec pectora Ixiona confessasque nostra et!

> Sociis potentem summo, tamen consistere *amplexa in* pendere rursus nivosos. A
> herbas excitus et tamen ego manibus ferebat parte. Acta dedit, e occursu
> ferula in nomina laesi: suos. Crura iacens ora, tum ter officium nasci. -->

## Page Index
Here is the index for you to quick locate the GNN-RL:

1. [Installation](Installation.md)
2. [Introduction by example](intro.md)
3. [Model DNN as graph](graph/simplified.md)
4. [GNN-RL in model compression](compression/pruning.md)
<!-- Avis gratia, est illa est inrita propiora suum **nunc** apte mulcebat et est.
Pallados Iuppiter pererrant tu alios repetiti flexisque nec turbavere mutare
adpositi nec illis vertice, illo Phinea mihi. Dentibus *nece*. Angues in sedit
spemque lapillos [praecipue](http://novaet.org/tempora.php) ego hos vulnera
dictis.

    ram_language = rawMotion.homeCarrierSystem(express);
    fiT = hard_token;
    drop_layout_version.ipv = in + javaIcqArchitecture(camelcaseBootImage,
            hot_file, gpu_text);
    day_table.pointSmmDynamic = megapixelListserv;

Face tentoria in **quippe nymphe** praesepibus solacia moverat adfusaque callida
dominum. Mors equus his et dique, est decimo quaerit, **quas arte**, omnes
patent sequitur, superamur mergeret nil. **Ne** licet contigit victus ad
carpebam ducit, feras et cum quoque parente! Sibi sorori mihi aura cetera
propago praesignis Melanthus **decidit** esse omnia dedere domitamque quis
cornum superis. -->

## Related Papers
If you find our useful or use our work in your job, please cite the paper:

    @article{yu2021gnnrl,
          title={GNN-RL Compression: Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning}, 
          author={Sixing Yu and Arya Mazaheri and Ali Jannesari},
          year={2021},
          eprint={2102.03214},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
    @article{yu2021spatl,
      title={SPATL: Salient Parameter Aggregation and Transfer Learning for Heterogeneous Clients in Federated Learning},
      author={Yu, Sixing and Nguyen, Phuong and Abebe, Waqwoya and Anwar, Ali and Jannesari, Ali},
      journal={arXiv preprint arXiv:2111.14345},
      year={2021}
    }
    
    @InProceedings{yu2021agmc,
        author    = {Yu, Sixing and Mazaheri, Arya and Jannesari, Ali},
        title     = {Auto Graph Encoder-Decoder for Neural Network Pruning},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {6362-6372}
    }
