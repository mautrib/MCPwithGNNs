from data.base import Base_Generator
from data.graph_generation import GENERATOR_FUNCTIONS
from toolbox.conversions import adjacency_matrix_to_tensor_representation, connectivity_to_dgl, dense_tensor_to_edge_format
from toolbox.searches.mcp import mcp_beam_method
import torch
import os
import toolbox.utils as utils
import random
import dgl
from networkx.algorithms.approximation.clique import max_clique
import tqdm
from toolbox.searches.mcp import MCP_Solver

class MCP_Generator(Base_Generator):
    """
    Generator for the Maximum Clique Problem.
    This generator plants a clique of 'clique_size' size in the graph.
    It is then used as a seed to find a possible bigger clique with this seed
    """
    def __init__(self, name, args):
        self.edge_density = args['edge_density']
        self.clique_size = int(args['clique_size'])
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'MCP_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size, 
                                                           self.edge_density)
        path_dataset = os.path.join(args['path_dataset'], 'mcp',
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)

    def _plant_clique(self, W):
        assert isinstance(self.clique_size, int), f"Clique size is not an int {self.clique_size=}"
        W, K = self.add_clique_base(W, self.clique_size)
        B = adjacency_matrix_to_tensor_representation(W)

        k_size = len(torch.where(K.sum(dim=-1)!=0)[0])
        seed = self.mcp_adj_to_ind(K)
        #K2 = mcp_beam_method(W,K,seeds=seed,add_singles=False) #Finds a probably better solution with beam search in the form of a list of indices. Will not work if k_size is 0 !
        #k2_size = len(K2)
        #if k2_size>k_size:
            #K = self.mcp_ind_to_adj(K2,self.n_vertices)
        return (B, K)
    
    def _get_max_clique(self, g, W):
        mc = max_clique(g)
        l_indices = [(id_i,id_j) for id_i in mc for id_j in mc if id_i!=id_j]
        t_ind = torch.tensor(l_indices)
        K = torch.zeros_like(W)
        K[t_ind[:,0],t_ind[:,1]] = 1

        B = adjacency_matrix_to_tensor_representation(W)
        return (B, K)


    def compute_example(self):
        """
        
        """
        try:
            g, W = GENERATOR_FUNCTIONS["ErdosRenyi"](self.edge_density, self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.clique_size==0:
            B,K = self._get_max_clique(g, W)
        else:
            B,K = self._plant_clique(W)
        K = K.to(int)
        K = self._solution_conversion(K, connectivity_to_dgl(B))
        return (B,K)
    
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
        
    @staticmethod
    def add_clique_base(W,k):
        K = torch.zeros((len(W),len(W)))
        K[:k,:k] = torch.ones((k,k)) - torch.eye(k)
        W[:k,:k] = torch.ones((k,k)) - torch.eye(k)
        return W, K

    @staticmethod
    def mcp_adj_to_ind(adj)->list:
        """
        adj should be of size (n,n) or (bs,n,n), supposedly the solution for mcp
        Transforms the adjacency matrices in a list of indices corresponding to the clique
        """
        solo=False
        if len(adj.shape)==2:
            solo=True
            adj = adj.unsqueeze(0)
        bs,_,_ = adj.shape
        sol_onehot = torch.sum(adj,dim=-1)#Gets the onehot encoding of the solution clique
        l_sol_indices = [torch.where(sol_onehot[i]!=0)[0] for i in range(bs)] #Converts the onehot encoding to a list of the nodes' numbers
        l_clique_sol = [{elt.item() for elt in indices} for indices in l_sol_indices]
        
        if solo:
            l_clique_sol=l_clique_sol[0]
        return l_clique_sol

    @staticmethod
    def mcp_ind_to_adj(ind,n)->torch.Tensor:
        """
        ind should be a set of indices (or iterable)
        Transforms it into the adjacency matrix of shape (n,n)
        """
        assert max(ind)<n, f"Index {max(ind)} not in range for {n} indices"
        adj = torch.zeros((n,n))
        n_indices = len(ind)
        x = [elt for elt in ind for _ in range(n_indices)]
        y = [elt for _ in range(n_indices) for elt in ind]
        adj[x,y] = 1
        adj *= (1-torch.eye(n))
        return adj

class MCP_Generator_True(Base_Generator):
    """
    Generator for the Maximum Clique Problem.
    This generator finds the max clique
    """
    def __init__(self, name, args):
        self.edge_density = args['edge_density']
        self.clique_size = int(args['clique_size'])
        self.n_threads = args.get('n_threads', 24)
        num_examples = args['num_examples_' + name]
        self.num_examples = num_examples
        self.n_vertices = args['n_vertices']
        subfolder_name = 'MCPTrue_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size,
                                                           self.edge_density)
        path_dataset = os.path.join(args['path_dataset'], 'mcptrue',
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)

    def create_dataset(self):
        l_adjacency = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            adj = self.compute_example()
            l_adjacency.append(adj)
        solver = MCP_Solver(l_adjacency, max_threads=self.n_threads)
        solver.solve()
        assert len(solver.solutions)==self.num_examples, "Error somewhere in MCP solver."
        clique_solutions = [solution[0] for solution in solver.solutions]
        l_b = [adjacency_matrix_to_tensor_representation(W) for W in l_adjacency]
        l_k = [MCP_Generator.mcp_ind_to_adj(elt, self.n_vertices) for elt in clique_solutions]
        l_k = [self._solution_conversion(K.to(int), connectivity_to_dgl(B)) for (B,K) in zip(l_b,l_k)]
        l_data = [(B,K) for (B,K) in zip(l_b,l_k)]
        return l_data

    def compute_example(self):
        try:
            g, W = GENERATOR_FUNCTIONS["ErdosRenyi"](self.edge_density, self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        assert isinstance(self.clique_size, int), f"Clique size is not an int {self.clique_size=}"
        if self.clique_size!=0:
            W, K = MCP_Generator.add_clique_base(W, self.clique_size)
        return W
    
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
