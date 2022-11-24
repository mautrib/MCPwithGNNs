"""
From DGL examples : https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/
With the edge version of : https://github.com/graphdeeplearning/benchmarking-gnns/

GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 n_classes,
                 n_layers,
                 activation = F.relu,
                 dropout = 0, **kwargs):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        #Embed input
        self.embedding_h = nn.Linear(in_features, hidden_features)
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_features, hidden_features, activation=activation))
        # output layer
        self.layers.append(GraphConv(hidden_features, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h = None):
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, depth_of_mlp=3):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(depth_of_mlp) ]
        list_FC_layers.append(nn.Linear( input_dim//2**depth_of_mlp , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.depth_of_mlp = depth_of_mlp
        
    def forward(self, x):
        y = x
        for l in range(self.depth_of_mlp):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.depth_of_mlp](y)
        return y

class GCN_Edge(nn.Module): #Same as above, but using a final step to apply
    def __init__(self,
                 in_features,
                 hidden_features,
                 n_classes,
                 n_layers,
                 depth_of_mlp=3,
                 activation = F.relu,
                 dropout = 0):
        super(GCN_Edge, self).__init__()
        self.layers = nn.ModuleList()
        #Embed input
        self.embedding_h = nn.Linear(in_features, hidden_features)
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(hidden_features, hidden_features, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        #For the output
        self.MLP_layer = MLPReadout(2 * hidden_features, n_classes, depth_of_mlp)

    def forward(self, g, h=None,e=None):
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.ndata['h'] = h
        g.apply_edges(_edge_feat)
        
        return g.edata['e']



