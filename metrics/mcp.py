import torch
import dgl
import numpy as np
from toolbox.conversions import dgl_dense_adjacency, edge_format_to_dense_tensor
from toolbox.searches.mcp import mcp_beam_method, mcp_beam_method_node
from metrics.common import edgefeat_total as common_edgefeat_total, fulledge_total as common_fulledge_total, node_total as common_node_total

###EDGEFEAT
def edgefeat_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_edges_i)
     - l_targets : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
     - l_adjacency: list of couples of tensors of size ((N_edges_i), (N_edges_i)) corresponding to the edges (src, dst) of the graph
    """

    l_dgl = [dgl.graph((src,dst)) for src,dst in l_adjacency]
    full_inferred = [edge_format_to_dense_tensor(inferred,graph) for inferred,graph in zip(l_inferred, l_dgl)]
    full_target = [edge_format_to_dense_tensor(target,graph) for target,graph in zip(l_targets, l_dgl)]
    full_adjacency = [dgl_dense_adjacency(graph) for graph in l_dgl]

    return fulledge_beamsearch(full_inferred, full_target, full_adjacency, beam_size=beam_size, suffix=suffix)

def edgefeat_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_edgefeat_total(l_inferred, l_targets))
    beam_sizes = [1280,500,100,10,1]
    for beam_size in beam_sizes:
        final_dict.update(edgefeat_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=beam_size, suffix=str(beam_size)))
    return final_dict

###FULLEDGE
def fulledge_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_i,N_i)
     - l_targets : list of tensors of shape (N_i,N_i)
     - l_adjacency: list of adjacency matrices of size (N_i,N_i)
    """
    assert len(l_inferred)==len(l_targets)==len(l_adjacency), f"Size of inferred, target and ajacency different : {len(l_inferred)}, {len(l_targets)} and {len(l_adjacency)}."
    bs = len(l_inferred)

    l_cliques = mcp_beam_method(l_adjacency, l_inferred, normalize=False, beam_size=beam_size)

    l_acc = []
    l_size_inf = []
    l_size_planted = []
    l_sep = []
    for inferred_clique, target in zip(l_cliques, l_targets):
        target_degrees = target.sum(-1)
        target_clique_set = set(torch.where(target_degrees>0)[0].detach().cpu().numpy())
        target_clique_size = len(target_clique_set)
        inf_clique_size = len(inferred_clique)
        
        true_pos = len(target_clique_set.intersection(inferred_clique))
        total_count = target_clique_size
        l_acc.append(float(true_pos/total_count))

        size_inf = inf_clique_size
        size_planted = target_clique_size
        size_error_percentage = (inf_clique_size-target_clique_size)/target_clique_size

        l_size_inf.append(float(size_inf))
        l_size_planted.append(float(size_planted))
        l_sep.append(float(size_error_percentage))


    assert np.all(np.array(l_acc))<=1, "Accuracy over 1, not normal."
    if suffix:
        suffix = '-' + suffix
    temp_d = {f'bs{suffix}-accuracy': l_acc, f'bs{suffix}-size_error_percentage': l_sep, f'bs{suffix}-size_inf': l_size_inf, f'bs{suffix}-size_planted': l_size_planted}
    final_dict = {}
    for key, value in temp_d.items():
        final_dict[key+'_std'] = np.std(value)
        final_dict[key] = np.mean(value)
    return final_dict

def fulledge_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_fulledge_total(l_inferred, l_targets))
    beam_sizes = [1280,500,100,10,1]
    for beam_size in beam_sizes:
        final_dict.update(fulledge_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=beam_size, suffix=str(beam_size)))
    return final_dict

## NODE

def node_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_nodes_i)
     - l_targets : list of tensors of shape (N_nodes_i) (For DGL, from target.ndata['solution'], for FGNN, converted)
     - l_adjacency: list of couples of tensors of size ((N_edges_i), (N_edges_i)) corresponding to the edges (src, dst) of the graph
    """
    assert len(l_inferred)==len(l_targets)==len(l_adjacency), f"Size of inferred, target and adjacency different : {len(l_inferred)}, {len(l_targets)} and {len(l_adjacency)}."
    bs = len(l_inferred)
    sizes = [(len(target),len(target)) for target in l_targets]
    base_adjs = [torch.zeros(size) for size in sizes]
    for base_adj,(src,dst) in zip(base_adjs,l_adjacency):
        base_adj[src,dst] = 1

    l_cliques = mcp_beam_method_node(base_adjs, l_inferred, normalize=False, beam_size=beam_size)

    l_acc = []
    l_size_inf = []
    l_size_planted = []
    l_sep = []
    for inferred_clique, target in zip(l_cliques, l_targets):
        target_clique_set = set(torch.where(target>0)[0].detach().cpu().numpy())
        target_clique_size = len(target_clique_set)
        inf_clique_size = len(inferred_clique)
        
        true_pos = len(target_clique_set.intersection(inferred_clique))
        total_count = target_clique_size
        l_acc.append(float(true_pos/total_count))

        size_inf = inf_clique_size
        size_planted = target_clique_size
        size_error_percentage = (inf_clique_size-target_clique_size)/target_clique_size

        l_size_inf.append(float(size_inf))
        l_size_planted.append(float(size_planted))
        l_sep.append(float(size_error_percentage))


    assert np.all(np.array(l_acc))<=1, "Accuracy over 1, not normal."
    if suffix:
        suffix = '-' + suffix
    temp_d = {f'bs{suffix}-accuracy': l_acc, f'bs{suffix}-size_error_percentage': l_sep, f'bs{suffix}-size_inf': l_size_inf, f'bs{suffix}-size_planted': l_size_planted}
    final_dict = {}
    for key, value in temp_d.items():
        final_dict[key+'_std'] = np.std(value)
        final_dict[key] = np.mean(value)
    return final_dict

def node_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_node_total(l_inferred, l_targets))
    beam_sizes = [1280,1]
    for beam_size in beam_sizes:
        final_dict.update(node_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=beam_size, suffix=str(beam_size)))
    return final_dict