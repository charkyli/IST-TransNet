from .network import *
from .context import *



def get_segmentation_model(name):
    if name == 'transnet_1':
        net = transnet(backbone='resnet34', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1)
    elif name == 'transnet_2':
        net = transnet(backbone='resnet18', scales=(10, 6, 5, 4), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1)
    else:
        raise NotImplementedError

    return net
