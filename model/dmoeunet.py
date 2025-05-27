# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import ensure_tuple_rep

from torch import Union
from typing import Tuple
from collections.abc import Sequence

class UnetOutUpBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, 
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, inp):
        return self.transp_conv(inp)


class ExpertContextUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: str,
        add_channels: int = 0,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels + add_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = UnetExpertResBlock(
            spatial_dims,
            out_channels + out_channels + add_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
    

class UnetExpertResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_name: str,
        act_name: tuple = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: str = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)

        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)

        out += residual
        out = self.lrelu(out)
        return out


# Define the Expert class
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


# Define the Mixture of Experts Layer class
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts=8, top_k=2):
        super(SparseMoE, self).__init__()
        self.router = nn.ModuleList([NoisyTopkRouter(n_embed, num_experts, top_k) for _ in range(5)]) # total attr num
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x, moe):
        router_out = [self.router[moe_](x_) for x_, moe_ in zip(x, moe)]
        gating_output = torch.stack([out for out, idx in router_out], dim=0)
        indices = torch.stack([idx for out, idx in router_out], dim=0)

        final_output = torch.zeros_like(x)
        attn_output = torch.zeros_like(x)[...,0]

        # Reshape inputs for batch processing
        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

            attn_output = None

        return final_output, attn_output, indices
    
    
# network
class ContextUNETR(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        feature_size: int = 24,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        args=None,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.regularizer = args.regularizer
        self.noise = args.moe
        self.dmoe = args.dmoe
        self.rag = args.rag
        self.stage = args.stage

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = in_channels,
            out_channels = feature_size,
            kernel_size =3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = 2 * feature_size ,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels= 2 * feature_size,
            out_channels = 4 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder10 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = 4 * feature_size,
            out_channels = 8 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)
        
        # decoder
        self.decoder4 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 8 ,
            out_channels = feature_size * 4,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

        self.decoder3 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 4,
            out_channels = feature_size * 2,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)
        
        self.decoder2 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 2,
            out_channels = feature_size,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

        self.decoder1 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = feature_size, 
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

        # out
        self.out = UnetOutUpBlock(spatial_dims=spatial_dims, 
            in_channels=feature_size, 
            out_channels=out_channels, 
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        
        feature_size_list = [self.encoder1.layer.conv3.out_channels, self.encoder2.layer.conv3.out_channels, self.encoder3.layer.conv3.out_channels, self.encoder4.layer.conv3.out_channels, self.encoder10.layer.conv3.out_channels]

        #  moe
        sparseMoE = []
        for sk_ch in feature_size_list: 
            sparseMoE.append(SparseMoE(sk_ch, num_experts=8, top_k=2)) 
        self.sparseMoE = nn.Sequential(*sparseMoE)
        
    def load_from(self, weights):
        pass

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def dMoE(self, current, attr, i):

        # dMoE
        ####################################
        bs, c, h, w, s = current.shape
        current = current.flatten(2).permute(0, 2, 1)

        # Apply the FFN
        sparseMoE_keys, attn, idx = self.sparseMoE[i](current, attr)
    
        # Add the residual connection (input to FFN) to the FFN output
        current = current + sparseMoE_keys

        current = current.transpose(1, 2).contiguous().view(bs, c, h, w, s)
        ####################################

        return current
    
    def forward(self, x_in, attr=None):

        hidden_states_out = []

        # 3D UNet
        enc0 = self.encoder1(x_in)
        enc0 = self.dMoE(enc0, attr, 0)

        enc1 = self.encoder2(enc0)
        enc1 = self.dMoE(enc1, attr, 1)

        enc2 = self.encoder3(enc1)
        enc2 = self.dMoE(enc2, attr, 2)

        enc3 = self.encoder4(enc2)
        enc3 = self.dMoE(enc3, attr, 3)

        dec4 = self.encoder10(enc3)
        dec4 = self.dMoE(dec4, attr, 4)
    
        hidden_states_out.append(enc0)
        hidden_states_out.append(enc1)
        hidden_states_out.append(enc2)
        hidden_states_out.append(enc3)
        hidden_states_out.append(dec4)

        dec2 = self.decoder4(hidden_states_out[4], hidden_states_out[3])
        dec1 = self.decoder3(dec2, hidden_states_out[2])
        dec0 = self.decoder2(dec1, hidden_states_out[1])
        out = self.decoder1(dec0, hidden_states_out[0])

        logits = self.out(out)

        return logits
      
