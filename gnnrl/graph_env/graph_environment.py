import os
import shutil
import torch
import torch.nn as nn
from torch.nn.utils import prune


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# from gnnrl.fine_tune import save_checkpoint
from gnnrl.graph_env.graph_construction import hierarchical_graph_construction, net_info
from gnnrl.graph_env.feedback_calculation import reward_caculation
from gnnrl.graph_env.flops_calculation import flops_caculation_forward, preserve_flops
# from gnnrl.graph_env.share_layers import share_layer_index
# from gnnrl.graph_env.network_pruning import channel_pruning


import numpy as np
import copy


class graph_env:

    def __init__(self,model,n_layer,dataset,val_loader,compression_ratio,g_in_size,log_dir,input_x,max_timesteps,model_name,device):
        #work space
        self.log_dir = log_dir
        self.device = device

        #DNN
        self.model = model
        self.model_name = model_name
        self.pruned_model = None
        #self.pruning_index = pruning_index
        self.input_x = input_x
        self.flops, self.flops_share = flops_caculation_forward(self.model, self.model_name, input_x, preserve_ratio=None)
        self.total_flops = sum(self.flops)
        print('---------------------------------------')
        print(self.flops)
        print(len(self.flops))
        print('---------------------------------------')
        print(self.flops_share)
        print('---------------------------------------')

        data = self.flops
        res_freq = stats.relfreq(data, numbins=20)

        pdf_value = res_freq.frequency

        cdf_value = np.cumsum(res_freq.frequency)
        x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
        # plt.bar(x, pdf_value, width=res_freq.binsize)
        plt.xlabel('参数量')
        plt.title(self.model_name)
        plt.plot(x, cdf_value)
        plt.show()
        # x = np.arange(1, len(self.flops)+1)


        # 绘制CDF
        # plt.figure(figsize=(10, 6))
        # plt.plot(x, data)
        # plt.title(self.model_name)
        # plt.xlabel('卷积层')
        # plt.ylabel('参数量')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        exit()

        self.in_channels,self.out_channels,_ = net_info(self.model_name)
        print(self.out_channels)

        self.preserve_in_c = copy.deepcopy(self.in_channels)
        self.preserve_out_c = copy.deepcopy(self.out_channels)
        print(self.preserve_out_c)
        self.pruned_out_c = None
        self.n_layer = n_layer
        #dataset
        self.dataset = dataset
        self.val_loader = val_loader

        #pruning
        self.desired_flops = self.total_flops * compression_ratio
        self.compression_ratio = compression_ratio
        self.preserve_ratio = torch.ones([n_layer])
        self.true_ratio = torch.ones([n_layer])
        self.best_accuracy = 0

        #graph
        self.g_in_size = g_in_size
        self.current_states = None
        #env
        self.done = False
        self.max_timesteps = max_timesteps
        _, accuracy,_,_ = reward_caculation(self.model, self.val_loader, self.device)
        print("Initial val. accuracy:",accuracy)

        # Save the original model state dict
        # print('Initial')
        ini_data = 'Initial val. accuracy:{}\nInitial FLOPs:{:.2f} GFLOPs'.format(accuracy,sum(self.flops)/1e9)
        with open(r"{}/{}_origin_data.txt".format(log_dir,self.model_name), 'w') as f:
            f.write(ini_data)
        # self.save_original_model_state_dict(origin_dir=self.log_dir)

    def reset(self):
        self.done=False
        self.pruned_model = None
        self.preserve_ratio = torch.ones([self.n_layer])
        self.true_ratio = torch.ones([self.n_layer])
        self.current_states = self.model_to_graph()
        self.preserve_in_c = copy.deepcopy(self.in_channels)
        self.preserve_out_c = copy.deepcopy(self.out_channels)
        self.pruned_out_c = None

        return self.current_states

    def step(self,actions,time_step):

        rewards = 0
        accuracy = 0

        # 剪枝策略搜索
        if not isinstance(self.model, nn.Module):
            print('Invalid input. Must be nn.Module')
            return
        else:
            self.pruned_model = copy.deepcopy(self.model)
            i = 0
            t = 0
            for name, module in self.pruned_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if self.model_name in ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet20']:
                        if i % 2 == 1:
                            self.preserve_ratio[i] *= 1 - actions[t].astype(float)
                            t += 1
                        else:
                            self.preserve_ratio[i] *= 1 - actions[0].astype(float)
                        # self.preserve_ratio = np.clip(self.preserve_ratio, 0.1, 1)
                    elif self.model_name in ['mobilenet']:
                        if module.groups == module.in_channels:
                            self.preserve_ratio[i] *= 1 - 0
                        else:
                            self.preserve_ratio[i] *= 1 - actions[t].astype(float)
                            t +=1
                    elif self.model_name in ['mobilenetv2', 'shufflenet', 'shufflenetv2']:
                        if module.groups == module.in_channels:
                            self.preserve_ratio[i] *= 1 - actions[t].astype(float)
                            t += 1
                        else:
                            self.preserve_ratio[i] *= 1 - 1
                    elif self.model_name == 'vgg16':
                        # self.preserve_ratio = actions
                        self.preserve_ratio[i] *= 1- actions[i].astype(float)
                    elif self.model_name in ['resnet18', 'resnet50']:
                        # self.preserve_ratio = actions
                        self.preserve_ratio[i] *= 1- actions[i].astype(float)
                    else:
                        # self.preserve_ratio = actions
                        self.preserve_ratio[i] *= 1- actions[i].astype(float)
                    if self.model_name in ['mobilenet', 'mobilenetv2']:
                        self.preserve_ratio = np.clip(self.preserve_ratio, 0.9, 1)
                    else:
                        self.preserve_ratio = np.clip(self.preserve_ratio, 0.1, 1)
                    prune.ln_structured(module, name='weight', amount=float(1 - self.preserve_ratio[i]), n=2, dim=0)
                    i += 1


        current_flops = preserve_flops(self.flops,self.preserve_ratio,self.model_name,actions)
        reduced_flops = self.total_flops - sum(current_flops)
        # ratio = 1 - reduced_flops/self.total_flops
        # print("reduced_flops: {};  The FLOPs ratio: {}".format(reduced_flops,ratio))
        #desired flops reduction

        if reduced_flops >= self.desired_flops:
            r_flops = 1 - reduced_flops/self.total_flops
            # print("FLOPS ratio:",r_flops)
            self.done = True
            if self.dataset == "cifar10":
                rewards, accuracy,_,_ = reward_caculation(self.pruned_model, self.val_loader, self.device )
            else:
                _,_,rewards, accuracy = reward_caculation(self.pruned_model, self.val_loader, self.device )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_checkpoint({
                    'model': self.model_name,
                    'dataset': self.dataset,
                    'preserve_ratio':self.preserve_ratio,
                    'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
                    'acc': self.best_accuracy,
                    'flops':r_flops
                }, True, checkpoint_dir=self.log_dir)

                print("Best Accuracy (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format( self.best_accuracy,r_flops))

                cur_data = ''
                if self.model_name == 'shufflenetv2':

                    print(f"current_flops_sum:{sum(current_flops) :.2f} FLOPs")
                # cur_data = 'best_accuracy: {} \nthe FLOPs ratio: {}'.format(self.best_accuracy,r_flops)
                    cur_data = f"best_accuracy: {self.best_accuracy} \nthe FLOPs ratio:{r_flops}\ncurrent_flops_sum:{sum(current_flops) :.2f} FLOPs"
                else:

                    print(f"current_flops_sum:{sum(current_flops) / 1e9:.2f} GFLOPs")
                    # cur_data = 'best_accuracy: {} \nthe FLOPs ratio: {}'.format(self.best_accuracy,r_flops)
                    cur_data = f"best_accuracy: {self.best_accuracy} \nthe FLOPs ratio:{r_flops}\ncurrent_flops_sum:{sum(current_flops) / 1e9:.2f} GFLOPs"
                with open(r"{}/{}_data.txt".format(self.log_dir,self.model_name), 'w') as f:
                    f.write(cur_data)


        if time_step == (self.max_timesteps):
            if not self.done:
                rewards = -100
                self.done = True

        self.pruned_channels()
        graph = self.model_to_graph()
        return graph,rewards,self.done


    def pruned_channels(self):
            self.preserve_in_c = copy.deepcopy(self.in_channels)
            self.preserve_in_c[1:] = (self.preserve_in_c[1:]*np.array(self.preserve_ratio[:-1]).reshape(-1)).astype(int)

            self.preserve_out_c = copy.deepcopy(self.out_channels)
            self.preserve_out_c = (self.preserve_out_c*np.array(self.preserve_ratio).reshape(-1)).astype(int)
            self.pruned_out_c = self.out_channels - self.preserve_out_c


    def model_to_graph(self):
        # from gnnrl.graph_env.graph_construction import hierarchical_graph_construction
        graph_data = hierarchical_graph_construction(
            self.preserve_in_c, self.preserve_out_c,
            self.model_name, self.g_in_size, self.device
        )
        # 新增全局池化处理确保固定维度
        for key in ['x', 'node_features', 'features']:
            if key in graph_data:
                # 对节点特征进行全局平均池化得到固定维度向量
                node_features = graph_data[key]
                state = torch.mean(node_features, dim=0)  # 形状变为 [D]
                return state
        for level_key in ['level1', 'level2']:
            if level_key in graph_data:
                level_data = graph_data[level_key]
                for key in ['x', 'node_features', 'features']:
                    if key in level_data:
                        node_features = level_data[key]
                        state = torch.mean(node_features, dim=0)  # 形状变为 [D]
                        return state
        available_keys = {k: list(graph_data[k].keys()) for k in graph_data if isinstance(graph_data[k], dict)}
        raise KeyError(f"Graph data missing required keys. Structure: {available_keys}. Expected 'x', 'node_features', or 'features'")

    def model_to_graph_plain(self):
        raise NotImplementedError

    def save_checkpoint(self,state, is_best, checkpoint_dir='.'):
        filename = os.path.join(checkpoint_dir, self.model_name+'ckpt.pth.tar')
        print('=> Saving checkpoint to {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            print('=> Saving best to {}'.format(filename))
            shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))

    # def save_original_model_state_dict(self, origin_dir='.'):
    #     filename = os.path.join(origin_dir, self.model_name + '_original.pth.tar')
    #     print('=> Saving original model to {}'.format(filename))
    #     torch.save({
    #         'model': self.model_name,
    #         'dataset': self.dataset,
    #         'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
    #         'flops': self.total_flops
    #     }, filename)



# class graph_env:
#
#     def __init__(self,model,n_layer,dataset,val_loader,compression_ratio,g_in_size,log_dir,input_x,max_timesteps,model_name,device):
#         #work space
#         self.log_dir = log_dir
#         self.device = device
#
#         #DNN
#         self.model = model
#         self.model_name = model_name
#         self.pruned_model = None
#         #self.pruning_index = pruning_index
#         self.input_x = input_x
#         self.flops, self.flops_share = flops_caculation_forward(self.model, self.model_name, input_x, preserve_ratio=None)
#         self.total_flops = sum(self.flops)
#         self.in_channels,self.out_channels,_ = net_info(self.model_name)
#
#         self.preserve_in_c = copy.deepcopy(self.in_channels)
#         self.preserve_out_c = copy.deepcopy(self.out_channels)
#         self.pruned_out_c = None
#         self.n_layer = n_layer
#         #dataset
#         self.dataset = dataset
#         self.val_loader = val_loader
#
#         #pruning
#         self.desired_flops = self.total_flops * compression_ratio
#         self.preserve_ratio = torch.ones([n_layer])
#         self.best_accuracy = 0
#
#         #graph
#         self.g_in_size = g_in_size
#         self.current_states = None
#         #env
#         self.done = False
#         self.max_timesteps = max_timesteps
#         _, accuracy,_,_ = reward_caculation(self.model, self.val_loader, self.device)
#         print("Initial val. accuracy:",accuracy)
#
#         # Save the original model state dict
#         # print("Saving origin module to {}".format(self.model_name + '_original.pth.tar'))
#         # self.save_original_model_state_dict(origin_dir=self.log_dir)
#
#     def reset(self):
#         self.done=False
#         self.pruned_model = None
#         self.preserve_ratio = torch.ones([self.n_layer])
#         self.current_states = self.model_to_graph()
#         self.preserve_in_c = copy.deepcopy(self.in_channels)
#         self.preserve_out_c = copy.deepcopy(self.out_channels)
#         self.pruned_out_c = None
#
#         return self.current_states
#
#     def step(self,actions,time_step):
#
#         rewards = 0
#         accuracy = 0
#         self.preserve_ratio *= 1 - np.array(share_layer_index(self.model,actions,self.model_name)).astype(float)
#         # self.preserve_ratio *= 1 - np.array(share_layer_index(self.model,actions,self.args.model)).astype(float)
#
#         if self.model_name in ['mobilenet','mobilenetv2']:
#             self.preserve_ratio = np.clip(self.preserve_ratio, 0.9, 1)
#         else:
#             self.preserve_ratio = np.clip(self.preserve_ratio, 0.1, 1)
#         # self.preserve_ratio = np.clip(self.preserve_ratio, 0.1, 0.98)
#
#         #pruning the model
#         # self.preserve_ratio[0] = 1
#         # self.preserve_ratio[-1] = 1
#         self.pruned_channels()
#
#         current_flops = preserve_flops(self.flops,self.preserve_ratio,self.model_name,actions)
#         reduced_flops = self.total_flops - sum(current_flops)
#
#         #desired flops reduction
#
#         if reduced_flops >= self.desired_flops:
#             r_flops = 1 - reduced_flops/self.total_flops
#             # print("FLOPS ratio:",r_flops)
#             self.done = True
#             self.pruned_model = channel_pruning(self.model,self.preserve_ratio)
#             if self.dataset == "cifar10":
#                 rewards, accuracy,_,_ = reward_caculation(self.pruned_model, self.val_loader, self.device )
#             else:
#                 _,_,rewards, accuracy = reward_caculation(self.pruned_model, self.val_loader, self.device )
#
#             if accuracy > self.best_accuracy:
#                 self.best_accuracy = accuracy
#
#                 self.save_checkpoint({
#                     'model': self.model_name,
#                     'dataset': self.dataset,
#                     'preserve_ratio':self.preserve_ratio,
#                     'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
#                     'acc': self.best_accuracy,
#                     'flops':r_flops
#                 }, True, checkpoint_dir=self.log_dir)
#
#                 print("Best Accuracy (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format( self.best_accuracy,r_flops))
#         if time_step == (self.max_timesteps):
#             if not self.done:
#                 rewards = -100
#                 self.done = True
#         graph = self.model_to_graph()
#         return graph,rewards,self.done
#
#
#     def pruned_channels(self):
#         self.preserve_in_c = copy.deepcopy(self.in_channels)
#         self.preserve_in_c[1:] = (self.preserve_in_c[1:]*np.array(self.preserve_ratio[:-1]).reshape(-1)).astype(int)
#
#         self.preserve_out_c = copy.deepcopy(self.out_channels)
#         self.preserve_out_c = (self.preserve_out_c*np.array(self.preserve_ratio).reshape(-1)).astype(int)
#         self.pruned_out_c = self.out_channels - self.preserve_out_c
#
#     def model_to_graph(self):
#         graph = hierarchical_graph_construction(self.preserve_in_c,self.preserve_out_c,self.model_name,self.g_in_size,self.device)
#         return graph
#
#     def model_to_graph_plain(self):
#         raise NotImplementedError
#
#     def save_checkpoint(self,state, is_best, checkpoint_dir='.'):
#         filename = os.path.join(checkpoint_dir, self.model_name+'ckpt.pth.tar')
#         print('=> Saving checkpoint to {}'.format(filename))
#         torch.save(state, filename)
#         if is_best:
#             print('=> Saving best to {}'.format(filename))
#             shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))
#
#     def save_original_model_state_dict(self, origin_dir='.'):
#         filename = os.path.join(origin_dir, self.model_name + '_original.pth.tar')
#         print('=> Saving original model to {}'.format(filename))
#         torch.save(self.model.state_dict(), filename)
