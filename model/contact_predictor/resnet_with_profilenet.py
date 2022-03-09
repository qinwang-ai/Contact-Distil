import torch
from torch import nn
from ..downstream_module import DownStreamModule
import torch.nn.functional as F

from .resnet_2d import ResNet2d


# Profile Net
class StretchNet(nn.Module):
    def __init__(self, dropout):
        super(StretchNet, self).__init__()
        self.bilstm = nn.LSTM(20, 128, num_layers=1, dropout=dropout,
                              bidirectional=True, bias=True)
        self.fc = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, profile, MSA_C=None):
        p, _ = self.bilstm(profile.transpose(0, 1))
        p = F.relu(p.transpose(0, 1))
        p = self.bn1(p.transpose(1, 2)).transpose(1, 2)
        p = self.fc(p)
        p = self.bn2(p.transpose(1, 2)).transpose(1, 2)
        N = F.sigmoid(p)

        profile = (1 - profile ** N) / N
        return profile, N

class ResnetWithProfileNet(ResNet2d):
    """
    contact predictor with resnet with profile net
    """
    def __init__(self, backbone_args, backbone_alphabet, num_classes, depth_reduction="mean", with_profilenet=True):
        """
        :param depth_reduction: mean, first
        """
        super().__init__(backbone_args, backbone_alphabet, depth_reduction)
        self.num_classes = num_classes
        self.with_profilenet = with_profilenet

        if self.with_profilenet == True:
            self.profileNet = StretchNet(dropout=0.3)

        self.first_layer = nn.Sequential(
            nn.Conv2d(256 + 40, 64, kernel_size=1),
        )

    def forward(self, tokens, embeddings, profile):
        # 1. profile embedding
        if self.with_profilenet == True:
            profile_embedding, _ = self.profileNet(profile)
        else:
            profile_embedding = profile

        # 2. transformer embedding
        # remove auxiliary tokens
        embeddings, padding_masks = self.remove_aux_tokens_in_embeddings(tokens, embeddings)

        batch_size, depth, seqlen, hiddendim = embeddings.size()

        # reduction
        embeddings = self.msa_depth_reduction(embeddings, padding_masks)

        # pre reduction 768 -> 128
        embeddings = self.pre_layer(embeddings)

        # 3. concat
        embeddings = torch.cat([embeddings, profile_embedding], dim=-1)

        hiddendim = 128 + 20

        # 4. pairwise concat embedding
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3)
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2)

        # 5.
        out = self.first_layer(pairwise_concat_embedding)
        out = self.res_layers(out)
        out = self.final_layer(out)
        # contact_map = torch.sigmoid(out.squeeze(1))
        contact_map = out.permute(0, 2, 3, 1)

        return contact_map


