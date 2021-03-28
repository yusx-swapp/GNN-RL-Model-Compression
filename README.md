# GNN-RL-Model-Compression
GNN-RL Compression: Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning

## Dependencies

Current code base is tested under following environment:

1. Python   3.8
2. PyTorch  1.8.0 (cuda 11.1)
3. torchvision 0.7.0
4. [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) 1.6.1

## GNN-RL Channel Pruning
In this work, we compress DNNs by performing FLOPs-constrained channel pruning (prunes Conv filters) on Conv layers. The GNN-RL has been tested on over-parameterized and mobile-friendly DNNs with different datasets (CIFAR-10 and ImageNet).
### CIFAR-10
In this subsection, DNNs are trained on CIFAR-10 since we observe a positive correlation between the pre-fine-tune accuracy and the post-fine-tuning accuracy. Pruning policies that obtain higher validation accuracy correspondingly have higher fine-tuned accuracy. It enables us to predict final model accuracy without fine-tuning, which results in an efficient and faster policy exploration.


To search the strategy on ResNet-110 with channel pruning (filter pruning) on Conv layers, and prunes 40% FLOPs reduction, by running:
   ```
python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet110 --compression_ratio 0.4 --log_dir ./logs
   ```
To search the strategy on ResNet-56 with channel pruning (filter pruning) on Conv layers, and prunes 50% FLOPs reduction, by running:
  ```
python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --log_dir ./logs
   ```

### ImageNet (ILSVRC-2012)
To evaluate the GNN-RL on the ImageNet (ILSVRC-2012), you need to first download the dataset from [ImageNet](http://www.image-net.org/download-images) and export the data.

Since the validation accuracy on the ImageNet dataset is sensitive to the compression ratio,
with high compression ratios, the accuracy drops considerably without fine-tuning 
(in some cases, the pruned model without fine-tuning has less than 1% validation accuracy).
We highly recommend that you decompose the pruning into several stages. 
For instance, to obtain a 49% FLOPs model, prune the target DNN two times, each with 70% FLOPs constraint~(i.e., 70% FLOPs times 70% FLOPs = 49% FLOPs).

If you have enough GPU resources, we also recommend you enable fine-tuning process on each RL search episode to ensure that the GNN-RL gets a valuable reward.


To search the strategy on VGG-16 with channel pruning on convolutional layers and fine-grained pruning on dense layers, and prunes 80% FLOPs reduction on convolutional layers, by running:
   ```
python -W ignore gnnrl_network_pruning.py --dataset imagenet --model vgg16 --compression_ratio 0.8 --log_dir ./logs --data_root [your dataset dir] 
   ```
To search the strategy on MobileNet-V1 with channel pruning on convolutional layers and fine-grained pruning on dense layers, and prunes 25% FLOPs reduction on convolutional layers, by running:
   ```
python -W ignore gnnrl_network_pruning.py --dataset imagenet --model mobilenet --compression_ratio 0.25 --val_size 5000  --log_dir ./logs --data_root [your imagenet dataset dir]
   ```

### Pruning tools
We apply the PyTorch built-in [pruning tools](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#extending-torch-nn-utils-prune-with-custom-pruning-functions)
`torch.nn.utils.prune` to prune a given DNN. 
This package prunes a neural network by mask those pruned weights, and it does [not accelerate the
inference](https://github.com/pytorch/pytorch/issues/36214).
If you want to accelerate the inference or save memory, please discard those weights with zero-masks. 

We also provide functions for you to extract these weights. For example, the VGG-16, you can extract weights and evaluate it by run:

```
python gnnrl_real_pruning.py --dataset imagenet --model vgg16 --data_root [your imagenet data dir] --ckpt_path data/pretrained_models
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

## Evaluate the compressed Model
After searching, we can evaluate the compressed Model, which is saved on the default directory ```./logs```.
We also provide the pruned and fine-tuned model in the ``` data/pretrained_models``` for you to evaluate them.
If we want to evaluate the performance of compressed Models py running:
```angular2html
python -W ignore gnnrl_fine_tune.py \
    --model=[model name] \
    --dataset=cifar10 \
    --n_gpu=4 \
    --batch_size=256 \
    --n_worker=32 \
    --data_root=data/datasets \
    --ckpt_path=[pruned model checkpoint path] \
    --eval
     
   ```

## Results after fine-tuning
| Models                   | Compressed ratio | Top1 Acc (%) | Dataset |
| ------------------------ | ------------     | ------------ |------------|
| ResNet-110                | 50% FLOPs        | **94.31**   |CIFAR-10|
| ResNet-56                | 50% FLOPs        | **93.49**   |CIFAR-10|
| ResNet-44                | 50% FLOPs       | **93.23**   |CIFAR-10|
| MobileNet-v1                | 40% FLOPs       | **69.5**   |ImageNet|
| VGG-16                | 20% FLOPs       | **70.35**   |ImageNet|
