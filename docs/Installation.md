## Installation
GNN-RL is tested under the the following environment and dependencies:
***
**Python       >=3.6**  
**PyTorch      1.9.1 (CUDA 11.1)**  
**CUDA Toolkit 11.1**
***
GNN-RL supports multiple Graph libraries as backends, e.g., DGL, PyG.   
***
[**DGL (with CUDA 11.1)**](https://www.dgl.ai/pages/start.html)   
[**PyG (torch-geometric)**](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) 
***


## GNN-RL installation from pip
Before installing the GNN-RL, installing the required PyTorch through pip,
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
installing the required Graph libraries,
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html
```
Install the gnnrl,
```bash
pip install gnnrl
```

## GNN-RL installation from conda
Will release soon




[comment]: <> (Pip)

[comment]: <> (python=3.7)

[comment]: <> (PyTorch 1.10.1 with CUDA 10.2)

[comment]: <> (```bash)

[comment]: <> (conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge)

[comment]: <> (pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html)

[comment]: <> (```)

[comment]: <> (conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge)

[comment]: <> (pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html)

[comment]: <> (conda install pyg -c pyg -c conda-forge)

[comment]: <> (pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html)


[comment]: <> (conda install -c dglteam dgl-cuda11.1)

[comment]: <> (pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html)

[comment]: <> (Will release soon)

[comment]: <> (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/LAS/jannesar-lab/yusx/anaconda3/lib/)

[comment]: <> (## Install from pip)
