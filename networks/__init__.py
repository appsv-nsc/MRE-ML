import argparse
from torch.nn import init
from collections.abc import Iterable
from networks.unet2d_custom import unet, unet_wo_skip, unet_dense
networks_dict = {
    'unet': unet,
    'unet_wo_skip': unet_wo_skip,
    'unet_dense': unet_dense,
}


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_unet_model(network_name):
    #network_name example: autoencoder_unet_2d
    mre_in_channels = 5
    in_channel = mre_in_channels
    task_name = network_name.split('_')[0]
    if 'autoencoder' in task_name:
        num_classes = in_channel
    elif 'rot' in task_name:
        num_classes = 4
    else:
        num_classes = 2

    if network_name.endswith("_WoSkip"):
        generic_network_name = 'unet_wo_skip'
        normalization = 'linear'
    elif network_name.endswith("_dense"):
        generic_network_name = 'unet_dense'
        normalization = 'softmax' if task_name == 'rot' else None
    else:
        generic_network_name = 'unet'
        normalization = 'linear'

    print("Network class used: ", generic_network_name)

    model = networks_dict[generic_network_name](in_channels=in_channel, num_classes=num_classes, first_stage_fsizes=(3, 3),normalization=normalization)

    return model


def get_networks(args):
    network_name = args.network
    network = get_unet_model(network_name=network_name)
    init_weights(network, args.init_weight_type)
    return network


def reset_all_lrs(optimizer, new_lr=1e-4):
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    print(f"All LRs reset to {new_lr}")
    
def rewarm_lr_groups(optimizer, keywords):
    """Set LR of groups whose tag matches any keyword back to their base_lr."""
    tags = set(keywords)
    for g in optimizer.param_groups:
        if g.get("tag") in tags:
            g["lr"] = g.get("base_lr", g["lr"])

def set_freeze_by_keywords(model, keywords, freeze=True):
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(k.find(x) != -1 for x in keywords):
            print('changing %s' % k)
            v.requires_grad = not freeze


def freeze_by_keywords(model, keywords):
    print('****** freezing ******')
    set_freeze_by_keywords(model, keywords, True)


def unfreeze_by_keywords(model, keywords):
    print('****** unfreezing ******')
    set_freeze_by_keywords(model, keywords, False)


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            # print(param.name)
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.im_channel = 3
    args.class_num = 3
    args.init_weight_type = "kaiming"
    name_list = list(networks_dict.keys())
    for i in range(len(name_list)):
        args.network = name_list[i]
        model = get_networks(args)
        # print(model)
