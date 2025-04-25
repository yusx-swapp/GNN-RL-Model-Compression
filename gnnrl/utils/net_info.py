from torch import nn
from thop import profile



def get_num_hidden_layer(net,model_name):
    layer_share=0

    n_layer=0

    filter_counts = []

    if model_name in ['mobilenet']:
        #layer_share = len(list(net.module.features.named_children()))+1
        sum = 0
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    n_layer +=1
                else:
                    n_layer +=1

                    layer_share+=1

                num_filters = module.out_channels
                # print(f'layer:{n_layer}___num:{num_filters}')
                sum += num_filters
                filter_counts.append(num_filters)
        print('layer_share:{};n_layer:{}'.format(layer_share,n_layer))
        print(f'filter_counts:{filter_counts}-------{len(filter_counts)}')
        print(sum)
        # exit()
    elif model_name in ['mobilenetv2','shufflenet','shufflenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels:
                    n_layer +=1
                    layer_share+=1
                else:
                    n_layer +=1

    elif model_name in ['resnet18','resnet50']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer +=1
                layer_share+=1
                num_filters = module.out_channels
                # print(f'layer:{n_layer}___num:{num_filters}')


    elif model_name in ['resnet110','resnet56','resnet44','resnet32','resnet20']:
        layer_share+=len(list(net.module.layer1.named_children()))
        layer_share+=len(list(net.module.layer2.named_children()))
        layer_share+=len(list(net.module.layer3.named_children()))
        layer_share+=1
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer+=1


                num_filters = module.out_channels
                # print(f'layer:{n_layer}___num:{num_filters}')
                filter_counts.append(num_filters)

        print('layer_share:{};n_layer:{}'.format(layer_share,n_layer))

        print(f'filter_counts:{filter_counts}-------{len(filter_counts)}')
        # exit()

    elif model_name == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_share+=1
                num_filters = module.out_channels
                # print(f'layer:{layer_share}___num:{num_filters}')
                filter_counts.append(num_filters)
        n_layer = layer_share
        print('layer_share:{};n_layer:{}'.format(layer_share,n_layer))
        print(f'filter_counts:{filter_counts}-------{len(filter_counts)}')
        # exit()
    else:
        raise NotImplementedError
    return n_layer,layer_share
