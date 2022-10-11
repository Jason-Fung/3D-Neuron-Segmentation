# This scrip will contain the different models that are to be used for training

## TODO: Each model has different inputs, how do we want to handle that interms of the input parameters to func Model.
## especially focusing on the ResNet.

## Also as a note I have not done the ResNet section yet because its kinda diff from the others and need to sync with 
## jason before proceeding.

def Model(ModName, img_size, in_channels, out_channels):
    if ModName == "BasicUNet":
        return monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
        features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
        norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)

    elif ModName == "UNETR":
        return monai.networks.nets.UNETR(in_channels, out_channels, img_size, 
        feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, 
        pos_embed='conv', norm_name='instance', conv_block=True, res_block=True, 
        dropout_rate=0.0, spatial_dims=3, qkv_bias=False)

    elif ModName == "SwinUNETR":
        return monai.networks.nets.SwinUNETR(img_size, in_channels, out_channels, 
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), feature_size=24, norm_name='instance', 
        drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, normalize=True, use_checkpoint=False, 
        spatial_dims=3, downsample='merging')

    elif ModName == "VNet":
        return monai.networks.nets.VNet(spatial_dims=3, in_channels=1, out_channels=1, 
        act=('elu', {'inplace': True}), dropout_prob=0.5, dropout_dim=3, bias=False)

    elif ModName == "ResNet":
        return monai.networks.nets.ResNet(block, layers, block_inplanes, spatial_dims=3, 
        n_input_channels=3, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', 
        widen_factor=1.0, num_classes=400, feed_forward=True)

    else:
        return print("Invalid model name.")

'''if __name__=='__main__':
    Model("sfd") '''