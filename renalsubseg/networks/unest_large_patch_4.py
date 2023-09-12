
# limitations under the License.

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.blocks.unetr_block import UnetstrBasicBlock, UnetrPrUpBlock, UnetResBlock


from networks.unest_block import UNesTConvBlock, UNestUpBlock, UNesTBlock

from monai.networks.blocks import Convolution

from networks.nest_transformer_3D import NestTransformer3D

import pdb

class UNesT(nn.Module):
    """
    UNesT model implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [96, 96, 96],
        feature_size: int = 16,
        patch_size: int = 2,
        depths: Tuple[int, int, int] = [2, 2, 2],
        num_heads: Tuple[int, int, int] = [3, 6, 12],
        embed_dim: Tuple[int, int, int] = [128, 256, 512],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        # featResBlock: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        
        self.embed_dim = embed_dim

        self.nestViT = NestTransformer3D(
            img_size=96, 
            in_chans=1, 
            patch_size=patch_size, 
            num_levels=3, 
            embed_dims=embed_dim,                 
            num_heads=num_heads, 
            depths=depths, 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,                
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='', 
            weight_init='', 
            global_pool='avg',
        )

        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 3,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 6,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )

        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=12 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=24 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UNesTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],
            out_channels=feature_size * 48,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UNesTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 24,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 24,
            out_channels=feature_size * 12,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 12,
            out_channels=feature_size * 6,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 6,
            out_channels=feature_size * 3,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=48*feature_size,
            out_channels=96*feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 3, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

            
    def forward(self, x_in):

        x, hidden_states_out = self.nestViT(x_in)

        enc0 = self.encoder1(x_in) 
        x1 = hidden_states_out[0]  
        enc1 = self.encoder2(x1) 
        x2 = hidden_states_out[1] 
        enc2 = self.encoder3(x2) 
        x3 = hidden_states_out[2] 
        enc3 = self.encoder4(x3) 
        x4 = hidden_states_out[3] 
        enc4 = x4 
        dec4 = x 
        dec4 = self.encoder10(dec4) 
        dec3 = self.decoder5(dec4, enc4) 
        dec2 = self.decoder4(dec3, enc3) 
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1) 
        out = self.decoder1(dec0, enc0) #
        logits = self.out(out)
        return logits
