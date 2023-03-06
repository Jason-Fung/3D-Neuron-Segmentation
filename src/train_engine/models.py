from monai.networks.nets import UNet, UNETR, SwinUNETR, VNet, ResNet

def Build_Model(cfg: dict, num_classes: int):

    model_name = cfg['MODEL']['model_arch']
    output_chnl = num_classes

    if model_name == "UNet":
        norm_type = cfg['MODEL']['norm']
        dropout = cfg['MODEL']['dropout']
        input_chnl = cfg['MODEL']['input_dim']
        spatial_dim = cfg['MODEL']['spatial_dim']
        enc_dec_channels = cfg['MODEL']['channel_layers']
        conv_strides = cfg['MODEL']['strides']
        res_units = cfg['MODEL']['num_res_units']

        model = UNet(spatial_dims=spatial_dim,
                        in_channels = input_chnl,
                        out_channels = output_chnl,
                        channels = enc_dec_channels,
                        strides=conv_strides,
                        num_res_units=res_units,
                        norm = norm_type,
                        dropout = dropout)

        return model

    elif model_name == "UNETR":
        norm_type = cfg['MODEL']['norm']
        dropout = cfg['MODEL']['dropout']
        input_chnl = cfg['MODEL']['input_dim']
        spatial_dim = cfg['MODEL']['spatial_dim']
        feature_size = cfg['MODEL']['feature_size']
        hidden_size = cfg['MODEL']['hidden_size']
        num_heads = cfg['MODEL']['num_heads']
        mlp = cfg['MODEL']['mlp_dim']

        patch_x = cfg['DATASET']['x_patch']
        patch_y = cfg['DATASET']['y_patch']
        patch_z = cfg['DATASET']['z_patch']
        if spatial_dim == 3:
            patch_size = (patch_z, patch_y, patch_x)
        else:
            patch_size = (patch_y, patch_x)

        model = UNETR(in_channels = input_chnl, 
                        out_channels = output_chnl, 
                        img_size = patch_size, 
                        feature_size=feature_size, 
                        hidden_size=hidden_size, 
                        mlp_dim=mlp, 
                        num_heads=num_heads, 
                        norm_name=norm_type,
                        dropout_rate=dropout, 
                        spatial_dims=spatial_dim)
        
        return model

    elif model_name == "SwinUNETR":

        norm_type = cfg['MODEL']['norm']
        dropout = cfg['MODEL']['dropout']
        attention_drop_out = cfg['MODEL']['attn_dropout']
        dropout_path_rate = cfg['MODEL']['dropout_path_rate']
        input_chnl = cfg['MODEL']['input_dim']
        spatial_dim = cfg['MODEL']['spatial_dim']
        feature_size = cfg['MODEL']['feature_size']
        num_heads = cfg['MODEL']['num_heads']
        depths = cfg['MODEL']['depths']

        patch_x = cfg['DATASET']['x_patch']
        patch_y = cfg['DATASET']['y_patch']
        patch_z = cfg['DATASET']['z_patch']

        if spatial_dim == 3:
            patch_size = (patch_z, patch_y, patch_x)
        else:
            patch_size = (patch_y, patch_x)


        model = SwinUNETR(img_size = patch_size, 
                          in_channels = input_chnl, 
                          out_channels = output_chnl, 
                          depths = depths, 
                          num_heads = num_heads, 
                          feature_size = feature_size, 
                          norm_name=norm_type, 
                          drop_rate=dropout, 
                          attn_drop_rate=attention_drop_out, 
                          dropout_path_rate=dropout_path_rate, 
                          normalize=True, 
                          use_checkpoint=False, 
                          spatial_dims=3, 
                          downsample='merging')
        
        return model

    else:
        return print("Invalid model name.")