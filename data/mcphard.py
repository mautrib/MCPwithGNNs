import os
from data.base import Base_Generator
from data.mcp import MCP_Generator, MCP_Generator_True
from metrics.common import edgefeat_ROC_AUC
from metrics.preprocess import edgefeat_converter
from models.baselines.base import Edge_NodeDegree
from toolbox import utils
from copy import deepcopy
import dgl
from toolbox.conversions import dense_tensor_to_edge_format
import numpy as np
import tqdm

BASELINE = Edge_NodeDegree()
PREPROCESS = edgefeat_converter
METRIC = edgefeat_ROC_AUC

class MCP_Generator_Hard(Base_Generator):
    """
    Generator for the 'Hard' Planted Maximum Clique Problem.
    This generator plants a clique of 'clique_size' size in the graph.
    It is then used as a seed to find a possible bigger clique with this seed.
    Then it keeps 'self.proportion' of the hardest graphs according to the baseline
    """

    proportion=0.1

    def __init__(self, name, args):
        self.init_args = args
        self.edge_density = args['edge_density']
        self.clique_size = int(args['clique_size'])
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        self.proportion = args.get('proportion', self.proportion)
        subfolder_name = 'MCPHard_{}_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size, 
                                                           self.edge_density,
                                                           self.proportion)
        path_dataset = os.path.join(args['path_dataset'], 'mcp',
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)
    
    def create_dataset(self):
        g = MCP_Generator(self.name, self.init_args)
        g.load_dataset(use_dgl=False)
        l_data = deepcopy(g.data)
        g.load_dataset(use_dgl=True)
        l_scores = []
        model = BASELINE
        for data, target in tqdm.tqdm(g.data, desc='Calculating scores...'):
            result = model(data)
            l_inferred, l_targets, _ = PREPROCESS(result, target)
            dic = METRIC(l_inferred, l_targets)
            for key, value in dic.items():
                if not '_std' in key:
                    score = value
                    break
            l_scores.append(score)
        sorted_idx = np.argsort(l_scores)
        real_num_examples = int(self.num_examples*self.proportion)
        sorted_idx_reduced = sorted_idx[:real_num_examples]
        print(f'Score limit: {l_scores[sorted_idx[real_num_examples]]}/{l_scores[sorted_idx[real_num_examples+1]]}')
        l_data = [l_data[i] for i in sorted_idx_reduced]
        return l_data

    @classmethod
    def _solution_conversion(cls, target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        target_dgl = dgl.add_self_loop(target_dgl)
        edge_classif = dense_tensor_to_edge_format(target, target_dgl)
        node_classif = (target.sum(dim=-1)!=0).to(target.dtype) # Get a node classification of shape (N)
        node_classif = node_classif.unsqueeze(-1) # Modify it to size (N,1)
        target_dgl.edata['solution'] = edge_classif
        target_dgl.ndata['solution'] = node_classif
        return target_dgl

class MCP_Generator_True_Hard(Base_Generator):
    """
    Generator for the 'Hard' Maximum Clique Problem.
    This generator finds the max clique, then keeps 'self.proportion' of the hardest graphs according to the baseline
    """
    proportion=0.1 #Proportion of data kept

    def __init__(self, name, args):
        self.init_args = args
        self.edge_density = args['edge_density']
        self.clique_size = int(args['clique_size'])
        self.n_threads = args.get('n_threads', 24)
        self.proportion = args.get('proportion', self.proportion)
        num_examples = args['num_examples_' + name]
        self.num_examples = num_examples
        self.n_vertices = args['n_vertices']
        subfolder_name = 'MCPTrueHard_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size,
                                                           self.edge_density,
                                                           self.proportion)
        path_dataset = os.path.join(args['path_dataset'], 'mcptrue',
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)   

    def create_dataset(self):
        g = MCP_Generator_True(self.name, self.init_args)
        g.load_dataset(use_dgl=False)
        l_data = deepcopy(g.data)
        g.load_dataset(use_dgl=True)
        l_scores = []
        model = BASELINE
        for data, target in tqdm.tqdm(g.data, desc='Calculating scores...'):
            result = model(data)
            l_inferred, l_targets, _ = PREPROCESS(result, target)
            dic = METRIC(l_inferred, l_targets)
            for key, value in dic.items():
                if not '_std' in key:
                    score = value
                    break
            l_scores.append(score)
        sorted_idx = np.argsort(l_scores)
        real_num_examples = int(self.num_examples*self.proportion)
        sorted_idx_reduced = sorted_idx[:real_num_examples]
        print(f'Score limit: {l_scores[sorted_idx[real_num_examples]]}/{l_scores[sorted_idx[real_num_examples+1]]}')
        l_data = [l_data[i] for i in sorted_idx_reduced]
        return l_data
    
    @classmethod
    def _solution_conversion(cls, target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        target_dgl = dgl.add_self_loop(target_dgl)
        edge_classif = dense_tensor_to_edge_format(target, target_dgl)
        node_classif = (target.sum(dim=-1)!=0).to(target.dtype) # Get a node classification of shape (N)
        node_classif = node_classif.unsqueeze(-1) # Modify it to size (N,1)
        target_dgl.edata['solution'] = edge_classif
        target_dgl.ndata['solution'] = node_classif
        return target_dgl