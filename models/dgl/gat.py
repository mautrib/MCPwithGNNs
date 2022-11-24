"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self,
                 n_layers=2,
                 in_features=1,
                 hidden_features=8,
                 n_classes=2,
                 num_heads=8,
                 num_out_heads=1,
                 activation = "elu",
                 feat_drop=0.6,
                 attn_drop=0.6,
                 negative_slope=0.2,
                 residual=False,
                 input_embed=False):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        if activation=="relu":
            self.activation = F.relu
        elif activation=="elu":
            self.activation = F.elu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        
        self.input_embed = input_embed
        if self.input_embed:
            self.embedding_h = nn.Linear(in_features, hidden_features)
            in_features = hidden_features

        heads = ([num_heads] * (n_layers-1)) + [num_out_heads]
        if n_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_features, hidden_features, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, n_layers-1):
                # due to multi-head, the in_features = hidden_features * num_heads
                self.gat_layers.append(GATConv(
                    hidden_features * heads[l-1], hidden_features, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                hidden_features * heads[-2], n_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_features, n_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata['feat']
        
        if self.input_embed:
            h = self.embedding_h(h.float())
        
        for l in range(self.n_layers):
            h = self.gat_layers[l](g, h)
            h = h.flatten(1) if l != self.n_layers - 1 else h.mean(1)
        return h

