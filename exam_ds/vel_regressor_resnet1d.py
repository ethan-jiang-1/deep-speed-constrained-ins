try:
    from exam_ds.vel_model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
except:
    from vel_model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


class ResNet1D_18(ResNet1D):
    def __init__(self):
        _fc_config['fc_dim'] = 512
        super(ResNet1D_18, self).__init__(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)


class ResNet1D_50(ResNet1D):
    def __init__(self):
        _fc_config['fc_dim'] = 1024
        super(ResNet1D_50, self).__init__(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                                          base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)


class ResNet1D_101(ResNet1D):
    def __init__(self):
        _fc_config['fc_dim'] = 1024
        super(ResNet1D_101, self).__init__(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                                          base_plane=64, output_block=FCOutputModule, **_fc_config)

'''
def get_model(arch, _input_channel, _output_channel):
    network = None
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    return network
'''

VelRegressorResnet18 = ResNet1D_18
VelRegressorResnet50 = ResNet1D_50
VelRegressorResnet101 = ResNet1D_101
