# GNN-RL-Model-Compression
GNN-RL Compression: Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning

## Dependencies

Current code base is tested under following environment:

1. Python   3.8
2. PyTorch  1.8.0 (cuda 11.1)
3. torchvision 0.7.0
4. [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) 1.6.1

## Training GNN-RL

### Channel Pruning
To search the strategy on ResNet-56 with channel pruning (filter pruning) on Conv layers, and prunes 50% FLOPs reduction, by running:
  ```
python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.005 --dataset cifar10 --bsize 32 --model resnet56 --compression_ratio 0.5 --warmup 30 --pruning_method cp --val_size 1000 --train_episode 200 --data_root ./data --log_dir [your log dir]
   ```
To search the strategy on ResNet-110 with channel pruning (filter pruning) on Conv layers, and prunes 50% FLOPs reduction, by running:
   ```
python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.005 --dataset cifar10 --bsize 32 --model resnet110 --compression_ratio 0.5 --warmup 30 --pruning_method cp --val_size 1000 --train_episode 200 --data_root ./data --log_dir [your log dir]
   ```

### Fine-tuning
To fine-tune the pruned 50%FLOPs ResNet-110, by running:

```angular2html
python -W ignore gnnrl_fine_tune.py \
    --model=resnet110 \
    --dataset=cifar10 \
    --lr=0.005 \
    --n_gpu=4 \
    --batch_size=256 \
    --n_worker=32 \
    --lr_type=cos \
    --n_epoch=200 \
    --wd=4e-5 \
    --seed=2018 \
    --data_root=data/datasets \
    --ckpt_path=[pruned model checkpoint path] \
    --finetuning
```
### ImageNet
To evaluate the GNN-RL on the ImageNet dataset, you need to first download the dataset from [ImageNet](http://www.image-net.org/download-images) (ILSVRC-2012) and export the data.

To search the strategy on VGG-16 with channel pruning on convolutional layers, by running:

   ```
python gnnrl_network_pruning.py --dataset ImageNet --model vgg16 --compression_ratio 0.8 --data_root [your dataset dir] --output [your logs dir]
   ```

## Evaluate the compressed Model
After searching, we can evaluate the compressed Model, which is saved on the default directory ```./logs```.
For example, if we want to evaluate the performance of compressed Models py running:
   ```
python -W ignore gnnrl_fine_tune.py \
    --model=resnet110 \
    --dataset=cifar10 \
    --n_gpu=4 \
    --batch_size=256 \
    --n_worker=32 \
    --data_root=data/datasets \
    --ckpt_path=[pruned model checkpoint path] \
    --eval
     
   ```

## Results after fine-tuning
| Models                   | Compressed ratio | Top1 Acc (%) |
| ------------------------ | ------------     | ------------ |
| ResNet-110                | 50% FLOPs        | **94.31**         |
| ResNet-56                | 50% FLOPs        | **93.49**   |
| ResNet-44                | 50% FLLOPs       | **93.23**   |
