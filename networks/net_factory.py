from networks.unet import UNet
from networks.VNet import VNet, corf, corf2d


def net_factory(net_type="unet", in_chns=1, class_num=2, mode="train", **kwargs):
    net = None
    
    # 处理模型名称变体
    if net_type.startswith("corn2d"):
        net_type = "corn2d"
    elif net_type.startswith("corn"):
        net_type = "corn"
    elif net_type.startswith("unet"):
        net_type = "unet"
    elif net_type.startswith("vnet"):
        net_type = "vnet"
    
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "corn" and mode == "train":
        net = corf(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "corn" and mode == "test":
        net = corf(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, **kwargs).cuda()
    elif net_type == "corn2d" and mode == "train":
        net = corf2d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "corn2d" and mode == "test":
        net = corf2d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, **kwargs).cuda()
    else:
        raise ValueError(f"Unsupported network type: {net_type} with mode: {mode}")
    
    return net
