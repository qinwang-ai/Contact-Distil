# encoding: utf-8

from torch import nn
import torch

class DownStreamModule(nn.Module):
    """
    base contact predictor for msa
    """
    def __init__(self, backbone_args, backbone_alphabet, depth_reduction="none"):
        super().__init__()
        self.backbone_args = backbone_args
        self.backbone_alphabet = backbone_alphabet

        self.prepend_bos = self.backbone_alphabet.prepend_bos
        self.append_eos = self.backbone_alphabet.append_eos
        self.eos_idx = self.backbone_alphabet.eos_idx
        if self.append_eos and self.eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")

        self.embed_dim = self.backbone_args.embed_dim
        self.attention_heads = self.backbone_args.attention_heads

        self.depth_reduction = depth_reduction
        if self.depth_reduction == "attention":
            self.msa_embed_dim_in = self.embed_dim
            self.msa_embed_dim_out = self.embed_dim // self.attention_heads
            self.msa_q_proj = nn.Linear(self.msa_embed_dim_in, self.msa_embed_dim_out)
            self.msa_k_proj = nn.Linear(self.msa_embed_dim_in, self.msa_embed_dim_out)


    def remove_aux_tokens_in_attentions(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        return attentions

    def remove_aux_tokens_in_embeddings(self, tokens, embeddings):
        padding_masks = tokens.eq(self.backbone_alphabet.padding_idx)  # B, R, C
        embeddings = embeddings * (1 - padding_masks.unsqueeze(-1).type_as(embeddings))

        # remove eos token attentions
        if self.append_eos:
            raise Exception("Not Implement")
        # remove cls token attentions
        if self.prepend_bos:
            padding_masks = padding_masks[..., 1:]
            embeddings = embeddings[..., 1:, :]

        if not padding_masks.any():
            padding_masks = None

        return embeddings, padding_masks

    def msa_depth_reduction(self, embeddings, padding_masks):
        if self.depth_reduction == "first":
            embeddings = embeddings[:, 0, :, :]
        elif self.depth_reduction == "mean":
            embeddings = torch.mean(embeddings, dim=1)
        elif self.depth_reduction == "attention":
            msa_q = self.msa_q_proj(embeddings[:, 0, :, :])  # first query
            msa_k = self.msa_k_proj(embeddings)  # all keys
            if padding_masks is not None:
                # Zero out any padded aligned positions - this is important since
                # we take a sum across the alignment axis.
                msa_q = msa_q * (1 - padding_masks[:, 0, :].unsqueeze(-1).type_as(msa_q))
            depth_attn_weights = torch.einsum("bld,bjld->bj", msa_q, msa_k)
            depth_attn_weights = torch.softmax(depth_attn_weights, dim=1)
            embeddings = torch.sum(embeddings * depth_attn_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        else:
            raise Exception("Wrong Depth Reduction Type")

        return embeddings

