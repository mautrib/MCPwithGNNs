import torch
import dgl
from toolbox.conversions import edge_format_to_dense_tensor

"""
These functions will convert different type of embeddings to others.
For now the only types that are interchangeable are "edgefeat" (DGL sparse type) and "fulledge", the dense tensors.
Each function tries to recognize the type of data passed to it. If it doesn't, it raises assertion or NotImplementedErrors
"""

def fulledge_converter(raw_scores, target, data=None, **kwargs):
    if isinstance(target, dgl.DGLGraph):
        if len(raw_scores)==target.num_edges():
            proba = torch.softmax(raw_scores,dim=-1)
            proba_of_being_1 = proba[:,1]
            target.edata['inferred'] = proba_of_being_1
            unbatched_graphs = dgl.unbatch(target)
            l_inferred = [edge_format_to_dense_tensor(graph.edata['inferred'], graph) for graph in unbatched_graphs]
            l_targets = [edge_format_to_dense_tensor(graph.edata['solution'], graph).squeeze() for graph in unbatched_graphs]
            l_adjacency = [edge_format_to_dense_tensor(torch.ones(graph.num_edges()),graph) for graph in unbatched_graphs]
        else:
            raise NotImplementedError(f"Didn't implement Node->Full Edge converter")
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_inferred = [rs for rs in raw_scores]
        l_targets = [edge_format_to_dense_tensor(graph.edata['solution'], graph).squeeze() for graph in unbatched_graphs]
        l_adjacency = [a for a in adjacency]
    return l_inferred, l_targets, l_adjacency

def edgefeat_converter(raw_scores, target, data=None, **kwargs):
    if isinstance(target, dgl.DGLGraph):
        if len(raw_scores)==target.num_edges():
            proba = torch.softmax(raw_scores,dim=-1)
            proba_of_being_1 = proba[:,1]
            
            target.edata['inferred'] = proba_of_being_1
            unbatched_graphs = dgl.unbatch(target)
            l_inferred = [graph.edata['inferred'] for graph in unbatched_graphs]
            l_target = [graph.edata['solution'].squeeze() for graph in unbatched_graphs]
            l_adjacency = [graph.edges() for graph in unbatched_graphs]
        else:
            raise NotImplementedError(f"Didn't implement Node -> Edge Feat converter")
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_adjacency = [(torch.where(adj>0)) for adj in adjacency]
        l_inferred = [ graph[src,dst] for (graph,(src,dst)) in zip(raw_scores,l_adjacency)]
        l_target = [ graph[src,dst] for (graph,(src,dst)) in zip(target,l_adjacency)]
    return l_inferred, l_target, l_adjacency

def node_converter(raw_scores, target, data=None, **kwargs):
    if isinstance(target, dgl.DGLGraph):
        proba = torch.softmax(raw_scores,dim=-1)
        proba_of_being_1 = proba[:,1]
        
        target.ndata['inferred'] = proba_of_being_1
        unbatched_graphs = dgl.unbatch(target)
        l_inferred = [graph.ndata['inferred'] for graph in unbatched_graphs]
        l_target = [graph.ndata['solution'].squeeze() for graph in unbatched_graphs]
        l_adjacency = [graph.edges() for graph in unbatched_graphs]
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized. Should be (batch size, n_vertices, n_vertices, N_Features)"
        adjacency = data[:,:,:,1]
        l_adjacency = [(torch.where(adj>0)) for adj in adjacency]
        l_inferred = [raw_score.squeeze(-1) for raw_score in raw_scores]
        l_target = [cur_target for cur_target in target]
    return l_inferred, l_target, l_adjacency
    