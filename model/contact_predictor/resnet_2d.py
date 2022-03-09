import torch
from torch import nn
from ..downstream_module import DownStreamModule

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # x commented
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock2(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # x commented
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2d(DownStreamModule):
    """
    contact predictor reproduced from
    Rives et al_2020_Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences


    1.linear projector - dimension reduction L * 768 -> 128

    2.pairwise outer concat embedding of query sequence - L * 128 -> L * L * 256

    3. (optional) concat with other feature maps together. e.g. RaptorX (6), msa embedding covariance features (256)

    4. Resnet
    4-1. 1*1 Conv 256 or lager -> 128
    4-2. Maxout Layer (stride 2?) - ? unknown - dimension reduction - 128 -> 64
    4-3. residual blocks. (num 32) : BN - ReLU - Conv 3 * 3 (64 feature maps) - Dropout (0.3) - ReLU - Conv 3 * 3 (64 feature maps) dilation rate alternating 1,2,4

    5. Final Conv 3*3  64 -> 1 + Sigmoid

    I think it is just to enlarge the respective field.


    questions:
    Maxout - do not change the resolution?
    how to padding during the convolution.

    For Us


    """
    def __init__(self, backbone_args, backbone_alphabet, num_classes, depth_reduction="mean"):
        """
        :param depth_reduction: mean, first
        """
        super().__init__(backbone_args, backbone_alphabet, depth_reduction)
        self.embed_dim_in = self.backbone_args.embed_dim
        self.attention_heads = self.backbone_args.attention_heads
        self.embed_dim_out = 768 #256 #self.backbone_args.embed_dim // self.attention_heads

        self.num_classes = num_classes

        self.pre_layer = nn.Linear(768, 128)

        self.first_layer = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            #nn.ReLU(inplace=True),
        )

        self.num_res_layers = 32
        self.res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            self.res_layers.append(BasicBlock2(inplanes=64, planes=64, dilation=dilation))
        self.res_layers = nn.Sequential(*self.res_layers)

        self.final_layer = nn.Conv2d(64, 2, kernel_size=3, padding=1)


    def forward(self, tokens, embeddings):
        # remove auxiliary tokens
        embeddings, padding_masks = self.remove_aux_tokens_in_embeddings(tokens, embeddings)

        batch_size, depth, seqlen, hiddendim = embeddings.size()

        # reduction
        embeddings = self.msa_depth_reduction(embeddings, padding_masks)

        # pre reduction 768 -> 128
        embeddings = self.pre_layer(embeddings)
        hiddendim = 128

        # pairwise concat embedding
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3)
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2)

        out = self.first_layer(pairwise_concat_embedding)
        out = self.res_layers(out)
        out = self.final_layer(out)

        #contact_map = torch.einsum("bick,bjck->bijc", q, k)
        #contact_map = torch.sigmoid(out.squeeze(1))
        contact_map = out.permute(0, 2, 3, 1)  #

        return contact_map


