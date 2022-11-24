from models.dgl.gat import GAT
from models.dgl.gcn import GCN, GCN_Edge
from models.dgl.gin import GIN, GINEdge, GINEdgeSimple
from models.dgl_edge import DGL_Edge
from models.dgl_node import DGL_Node
from models.fgnn_edge import FGNN_Edge
from models.fgnn_node import FGNN_Node
from models.dgl.gatedgcn import GatedGCNNet_Edge, GatedGCNNet_Node
from models.fgnn.fgnn import Simple_Edge_Embedding, Simple_Node_Embedding, RS_Edge_Embedding, RS_Node_Embedding
from models.baselines.base import Edge_NodeDegree
import logging

DUMMY_MODEL_NAMES = ('node_degree',)

def is_dummy(name):
    return name in DUMMY_MODEL_NAMES

DUMMY_MODELS = {
    'node_degree': {'edge': Edge_NodeDegree}
}

FGNN_EMBEDDING_DICT = {
    'edge': FGNN_Edge,
    'node': FGNN_Node,
}

DGL_EMBEDDING_DICT = {
    'edge': DGL_Edge,
    'node': DGL_Node
}

MODULE_DICT = {
    'fgnn' : {  'edge': Simple_Edge_Embedding,
                'node': Simple_Node_Embedding},
    'rsfgnn' : {    'edge' : RS_Edge_Embedding,
                    'node' : RS_Node_Embedding},
    'gatedgcn' : {  'edge': GatedGCNNet_Edge,
                    'node': GatedGCNNet_Node    },
    'gcn': {    'edge' : GCN_Edge,
                'node' : GCN                    },
    'gin': {    'edge' : GINEdge,
                'node' : GIN                    },
    'ginsimple': {  'edge' : GINEdgeSimple,
                    'node' : GIN                },
    'gat': {    'node': GAT                     }
}

NOT_DGL_ARCHS = ('fgnn', 'rsfgnn')

def check_dgl_compatibility(use_dgl, arch_name, dgl_check=True):
    arch_uses_dgl = not(arch_name in NOT_DGL_ARCHS)
    warning_str=''
    if use_dgl and not(arch_uses_dgl):
        warning_str = f"Architecture '{arch_name}' is registered as not using DGL but you want it to. If it should use DGL, please remove '{arch_name}' from variable 'NOT_DGL_ARCHS' in models/__init__.py"
    elif not(use_dgl) and arch_uses_dgl:
        warning_str = f"Architecture '{arch_name}' is registered as using DGL but you're not using DGL. If it shouldn't use DGL, please add '{arch_name}' it to variable 'NOT_DGL_ARCHS' in models/__init__.py"
    if warning_str:
        if dgl_check:
            raise TypeError(warning_str)
        else:
            logging.exception(warning_str)

def get_torch_model(config):
    arch_name = config['arch']['name'].lower()
    embedding = config['arch']['embedding'].lower()
    Module_Class = MODULE_DICT[arch_name][embedding]
    module_config = config['arch']['configs'][arch_name]
    module = Module_Class(**module_config)
    return module

def get_optim_args(config):
    return config['train']['optim_args']

def get_dummy_pl_model(config, dgl_check=True):
    arch_name = config['arch']['name'].lower()
    embedding = config['arch']['embedding'].lower()
    use_dgl = config['arch']['use_dgl']
    check_dgl_compatibility(use_dgl, arch_name, dgl_check=dgl_check)
    PLModel = DUMMY_MODELS[arch_name][embedding]
    return PLModel

def get_pl_model(config, dgl_check=True):
    arch_name = config['arch']['name'].lower()
    embedding = config['arch']['embedding'].lower()
    use_dgl = config['arch']['use_dgl']
    check_dgl_compatibility(use_dgl, arch_name, dgl_check=dgl_check)
    if use_dgl:
        PL_Model = DGL_EMBEDDING_DICT[embedding]
    else:
        PL_Model = FGNN_EMBEDDING_DICT[embedding]
    return PL_Model

def get_dummy_pipeline(config, dgl_check=True):
    batch_size = config['train']['batch_size']
    PLModel = get_dummy_pl_model(config, dgl_check)
    pipeline = PLModel(batch_size=batch_size)
    return pipeline

def get_gnn_pipeline(config, dgl_check=True):
    PL_Model = get_pl_model(config, dgl_check)
    
    module = get_torch_model(config)

    optim_args = get_optim_args(config)

    batch_size = config['train']['batch_size']
    pipeline = PL_Model(module, optim_args, batch_size=batch_size)

    return pipeline

def get_pipeline(config, dgl_check=True):
    if config['arch']['name'] in DUMMY_MODEL_NAMES: 
        return get_dummy_pipeline(config, dgl_check)
    else:
        return get_gnn_pipeline(config, dgl_check)



