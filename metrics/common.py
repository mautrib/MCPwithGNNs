import torch
import sklearn.metrics as sk_metrics
import numpy as np

### EDGE FEAT

def edgefeat_compute_accuracy(l_inferred, l_targets) -> dict:
    """
     - raw_scores : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    l_acc = []
    for inferred,solution in zip(l_inferred, l_targets):
        n_solution_edges = torch.sum(solution)
        n_edges = len(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) 
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        true_cat = torch.sum(y_onehot*solution) + torch.sum((1-y_onehot)*(1-solution))
        acc = float(true_cat/n_edges)
        l_acc.append(acc)
    assert np.all(np.array(l_acc)<=1), "Accuracy over 1, not normal."
    return {'accuracy': np.mean(l_acc), 'accuracy_std': np.std(l_acc)}

def edgefeat_compute_f1(l_inferred, l_targets) -> dict:
    """
     - raw_scores : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    prec, rec = 0, 0
    l_prec, l_rec, l_f1 = [], [], []
    for inferred,solution in zip(l_inferred, l_targets):
        n_solution_edges = torch.sum(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) 
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        prec = float(sk_metrics.precision_score(solution.detach().cpu().numpy(), y_onehot.detach().cpu().numpy()))
        rec = float(sk_metrics.recall_score(solution.detach().cpu().numpy(), y_onehot.detach().cpu().numpy()))
        f1 = 0
        if prec+rec!=0:
            f1 = 2*(prec*rec)/(prec+rec)
        l_prec.append(prec)
        l_rec.append(rec)
        l_f1.append(f1)
    
    return {'precision': np.mean(l_prec), 'recall': np.mean(l_rec), 'f1': np.mean(l_f1), 'precision_std': np.std(l_prec), 'recall_std': np.std(l_rec), 'f1_std': np.std(l_f1)}

def edgefeat_ROC_AUC(l_inferred, l_targets) -> dict:
    """
     - inferred : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    l_auc = []
    for inferred, target in zip(l_inferred, l_targets):
        auc = float(sk_metrics.roc_auc_score(target.detach().cpu().numpy(), inferred.detach().cpu().numpy()))
        l_auc.append(auc)
    return {'roc_auc': np.mean(l_auc), 'roc_auc_std': np.std(l_auc)}

def edgefeat_PR_AUC(l_inferred, l_targets) -> dict:
    """
     - inferred : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    l_auc = []
    for inferred, target in zip(l_inferred, l_targets):
        precision, recall, _ = sk_metrics.precision_recall_curve(target.detach().cpu().numpy(), inferred.detach().cpu().numpy())
        auc = sk_metrics.auc(recall, precision)
        auc = float(auc)
        l_auc.append(auc)
    return {'pr_auc': np.mean(l_auc), 'pr_auc_std': np.std(l_auc)}

def edgefeat_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(edgefeat_ROC_AUC(l_inferred, l_targets))
    final_dict.update(edgefeat_PR_AUC(l_inferred, l_targets))
    final_dict.update(edgefeat_compute_accuracy(l_inferred, l_targets))
    final_dict.update(edgefeat_compute_f1(l_inferred, l_targets))
    return final_dict

### FULL EDGE

def fulledge_compute_accuracy(l_inferred, l_targets):
    """
     - raw_scores : list of tensors of shape (N,N)
     - target : list of tensors of shape (N,N) (For DGL, for FGNN, the target, for DGL, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {l_inferred.shape} and {len(l_targets.shape)}."
    bs = len(l_inferred)
    l_acc = []
    for cur_inferred, cur_target in zip(l_inferred, l_targets):
        cur_inferred = cur_inferred.flatten()
        cur_target = cur_target.flatten()
        n_solution_edges = cur_target.sum()
        n_edges = len(cur_target)
        _,ind = torch.topk(cur_inferred, k=n_solution_edges)
        y_onehot = torch.zeros_like(cur_inferred)
        y_onehot = y_onehot.type_as(cur_target)
        y_onehot.scatter_(0, ind, 1)
        acc = float((torch.sum(y_onehot*cur_target) + torch.sum((1-y_onehot)*(1-cur_target)))/(n_edges))
        l_acc.append(acc)
    assert np.all(np.array(l_acc)<=1), "Accuracy over 1, not normal."
    return {'accuracy': np.mean(l_acc), 'accuracy_std': np.std(l_acc)}

def fulledge_compute_f1(l_inferred, l_targets):
    """
     - raw_scores : list of tensors of shape (N,N)
     - target : list of tensors of shape (N,N) (For DGL, for FGNN, the target, for DGL, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {l_inferred.shape} and {len(l_targets.shape)}."
    bs = len(l_inferred)
    l_prec, l_rec, l_f1 = [], [], []
    for cur_inferred, cur_target in zip(l_inferred, l_targets):
        cur_inferred = cur_inferred.flatten()
        cur_target = cur_target.flatten()
        n_solution_edges = cur_target.sum()
        _,ind = torch.topk(cur_inferred, k=n_solution_edges)
        y_onehot = torch.zeros_like(cur_inferred)
        y_onehot = y_onehot.type_as(cur_target)
        y_onehot.scatter_(0, ind, 1)

        prec = sk_metrics.precision_score(cur_target.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
        rec  = sk_metrics.recall_score(cur_target.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
        f1 = 0
        if prec+rec!=0:
            f1 = 2*(prec*rec)/(prec+rec)
        l_prec.append(prec)
        l_rec.append(rec)
        l_f1.append(f1)
    return {'precision': np.mean(l_prec), 'recall': np.mean(l_rec), 'f1': np.mean(l_f1), 'precision_std': np.std(l_prec), 'recall_std': np.std(l_rec), 'f1_std': np.std(l_f1)}
        
def fulledge_ROC_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensor of shape (N_i, N_i)
     - l_targets : list of tensors of shape (N__i, N_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    l_auc = []
    for inferred, target in zip(l_inferred, l_targets):
        auc = float(sk_metrics.roc_auc_score(target.detach().cpu().numpy().flatten(), inferred.detach().cpu().to(int).numpy().flatten()))
        l_auc.append(auc)
    return {'roc_auc': np.mean(l_auc), 'roc_auc_std': np.std(l_auc)}

def fulledge_PR_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensor of shape (N_i, N_i)
     - l_targets : list of tensors of shape (N__i, N_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    l_auc = []
    for inferred, target in zip(l_inferred, l_targets):
        precision, recall, _ = sk_metrics.precision_recall_curve(target.detach().cpu().numpy(), inferred.detach().cpu().numpy())
        auc = sk_metrics.auc(recall, precision)
        auc = float(auc)
    return {'pr_auc': np.mean(l_auc), 'pr_auc_std': np.std(l_auc)}

def fulledge_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(fulledge_ROC_AUC(l_inferred, l_targets))
    final_dict.update(fulledge_PR_AUC(l_inferred, l_targets))
    final_dict.update(fulledge_compute_accuracy(l_inferred, l_targets))
    final_dict.update(fulledge_compute_f1(l_inferred, l_targets))
    return final_dict

### NODE

def node_compute_accuracy(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_compute_accuracy(l_inferred, l_targets)

def node_compute_f1(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_compute_f1(l_inferred, l_targets)

def node_ROC_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_ROC_AUC(l_inferred, l_targets)

def node_PR_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_PR_AUC(l_inferred, l_targets)

def node_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(node_ROC_AUC(l_inferred, l_targets))
    final_dict.update(node_PR_AUC(l_inferred, l_targets))
    final_dict.update(node_compute_accuracy(l_inferred, l_targets))
    final_dict.update(node_compute_f1(l_inferred, l_targets))
    return final_dict




