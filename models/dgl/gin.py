"""
From DGL examples : https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/

GIN using DGL nn package
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from copy import deepcopy

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, depth_of_mlp, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        input_dim: int
            The dimensionality of input features
        output_dim: int
            The number of classes for prediction
        depth_of_mlp: int
            The number of linear layers
        hidden_dim: int
            The dimensionality of hidden_layers

        """
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.linear_or_not = True  # default is linear model
        self.num_layers = depth_of_mlp
        self.output_dim = output_dim

        if depth_of_mlp < 1:
            raise ValueError("number of layers should be positive!")
        elif depth_of_mlp == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(depth_of_mlp - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(depth_of_mlp - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, n_layers, depth_of_mlp, in_features, hidden_features,
                 n_classes, final_dropout=0., learn_eps=False, graph_pooling_type='sum',
                 neighbor_pooling_type='mean'):
        """model parameters setting

        Parameters
        ---------
        n_layers: int
            The number of linear layers in the neural network
        depth_of_mlp: int
            The number of linear layers in mlps
        in_features: int
            The dimensionality of input features
        hidden_features: int
            The dimensionality of hidden units at ALL layers
        n_classes: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps

        self.embedding_h = nn.Linear(in_features, hidden_features)

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            mlp = MLP(depth_of_mlp, hidden_features, hidden_features, hidden_features)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.linears_prediction.append(
                    nn.Linear(hidden_features, n_classes)) #hidden_features))#

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h.float())
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            score_over_layer += self.drop(self.linears_prediction[i](h))

        return score_over_layer 


class GINEdgeSimple(nn.Module):
    """GIN model with an edge embedding at the end."""
    def __init__(self, *args, **kwargs):
        super(GINEdgeSimple, self).__init__()
        depth_of_mlp = kwargs['depth_of_mlp']
        hidden_features = kwargs['hidden_features']
        n_classes = kwargs['n_classes']
        gin_node_dict = deepcopy(kwargs)
        gin_node_dict['n_classes'] = hidden_features
        self.gin = GIN(*args, **gin_node_dict)
        self.MLP_layer = MLP(depth_of_mlp, 2 * hidden_features, hidden_features, n_classes)
    
    def forward(self, g, h=None, e=None):
        if h is None:
            h = g.ndata['feat']
        h = self.gin(g,h)
        g.ndata['h'] = h

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e']
    
class GINEdge(nn.Module):
    """GIN Edge model inspired by : https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/TSP_edge_classification/gin_net.py"""
    def __init__(self, n_layers, depth_of_mlp, in_features, hidden_features,
                 n_classes, final_dropout=0., learn_eps=False, graph_pooling_type='sum',
                 neighbor_pooling_type='sum'):
        super(GINEdge, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps

        self.embedding_h = nn.Linear(in_features, hidden_features)

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            mlp = MLP(depth_of_mlp, hidden_features, hidden_features, hidden_features)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(n_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(in_features, n_classes))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_features, n_classes))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        # Non-linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.prediction.append(
                nn.Sequential(
                    nn.Linear(2*hidden_features, hidden_features),
                    nn.ReLU(),
                    nn.Linear(hidden_features, n_classes)
                )
            )

    def forward(self, g, h=None, e=None):
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            return {'e': e}
        
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h.float())
        g.ndata['h'] = h
        g.apply_edges(_edge_feat)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [g.edata['e']]
        
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            g.ndata['h'] = h
            g.apply_edges(_edge_feat)
            hidden_rep.append(g.edata['e'])

        score_over_layer = 0
        for i, e in enumerate(hidden_rep):
            score_over_layer += self.prediction[i](e)
        
        return score_over_layer
