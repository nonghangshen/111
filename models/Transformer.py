import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import numpy as np

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.activation = self.get_activation_fn(activation)

    def get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                pos_q: Optional[Tensor] = None, pos_kv: Optional[Tensor] = None) -> Tensor:
        # Ensure query, key, and value are reshaped properly to [L, B, C]
        if query.dim() == 4:  # [B, C, H, W] shape, need to reshape
            B, C, H, W = query.shape
            query = query.view(B, C, H * W).permute(2, 0, 1)  # [L, B, C]
            key = key.view(B, C, H * W).permute(2, 0, 1)
            value = value.view(B, C, H * W).permute(2, 0, 1)

        if pos_q is not None:
            query = query + pos_q
        if pos_kv is not None:
            key = key + pos_kv
            value = value + pos_kv

        attn_output, _ = self.attn(query=query, key=key, value=value, key_padding_mask=key_padding_mask)

        # Concatenate query and attention output
        concat = torch.cat([query, attn_output], dim=-1)
        fused = self.fusion_proj(concat)
        output = self.norm(self.dropout(fused))  # [L, B, C]
        output = self.activation(output)
        return output

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = self.get_activation_fn(activation)

    def get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        # If x is 4D [B, C, H, W], flatten it to 3D [B, L, C]
        if x.dim() == 4:  # [B, C, H, W] shape
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, L, C] here L = H * W

        # Add position embedding if provided
        if pos is not None:
            x = x + pos

        # Perform self-attention
        x = x.permute(1, 0, 2)  # Convert to [L, B, C] for MultiheadAttention
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        output = x + self.dropout(attn_output)  # Residual connection
        output = output.permute(1, 0, 2)  # Convert back to [B, L, C]

        # Apply activation function
        output = self.activation(output)

        return self.norm(output)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1,
                 activation="relu", return_intermediate_dec=False, args=None):
        super().__init__()

        self.args = args
        self.self_attn = nn.ModuleList([
            SelfAttention(d_model=d_model, nhead=args.attn_heads, dropout=args.attn_dropout, activation=activation)
            for _ in range(args.self_attn_layers)
        ])

        self.cross_attn = nn.ModuleList([
            CrossAttention(d_model=d_model, nhead=args.attn_heads, dropout=args.attn_dropout, activation=activation)
            for _ in range(args.cross_attn_layers)
        ])

    def forward(self, content, style, mask, pos_embed_c=None, pos_embed_s=None):
        hs = content
        B, C, H, W = content.shape  # Extract height and width here for reshaping
        for layer in self.self_attn:
            hs = layer(hs, key_padding_mask=mask, pos=pos_embed_c)

        for layer in self.cross_attn:
            hs = layer(hs, style, style, pos_q=pos_embed_c, pos_kv=pos_embed_s)
        return hs


