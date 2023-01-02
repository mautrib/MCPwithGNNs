from torch.utils.data import DataLoader
from dgl import batch as dglbatch
from data.mcphard import MCP_Generator_Hard, MCP_Generator_True_Hard
from models import check_dgl_compatibility
from toolbox.conversions import edge_format_to_dense_tensor
import toolbox.maskedtensor as maskedtensor
import logging
import torch

from data.mcp import MCP_Generator, MCP_Generator_True

TRAIN_VAL_TEST_LOOKUP = {
    'train': 'train',
    'val': 'train',
    'test': 'test'
} #If we're validating or testing, we'll check the config under the 'train' key. For testing, it's 'test'

MASKEDTENSOR_PROBLEMS = []

EMBED_TYPES = {
    'rsnode': 'node',
    'rsedge': 'edge'
}

def get_generator_class(problem_key):
    if problem_key == 'mcp':
        return MCP_Generator
    elif problem_key == 'mcptrue':
        return MCP_Generator_True
    elif problem_key == 'mcphard':
        return MCP_Generator_Hard
    elif problem_key == 'mcptruehard':
        return MCP_Generator_True_Hard
    raise NotImplementedError(f"Generator for problem {problem_key} hasn't been implemented or defined in data/__init__.py yet.")

def check_maskedtensor_compatibility(use_maskedtensor, problem, force_check=True):
    """
    In case there is a need for using masked tensor, which enable the use of variable size of graphs with FGNNs.
    """
    problem_uses_mt = problem in MASKEDTENSOR_PROBLEMS
    warning_str=''
    if use_maskedtensor and not(problem_uses_mt):
        warning_str = f"Problem '{problem}' is not registered as using masked tensors but you want it to. " + \
                      f"If it should use masked tensors, please add '{problem}' in variable 'MASKEDTENSOR_PROBLEMS' in data/__init__.py"
    elif not(use_maskedtensor) and problem_uses_mt:
        warning_str = f"Problem '{problem}' is registered as using masked tensors but you don't want it to. " + \
                      f"If it should'nt use masked tensors, please remove '{problem}' from variable 'MASKEDTENSOR_PROBLEMS' in data/__init__.py"
    if warning_str:
        if force_check:
            raise TypeError(warning_str)
        else:
            logging.exception(warning_str)

def _collate_fn_tensor_node(samples_list):
    """
    Custom collate function for FGNN dense data with node embeddings.
    """
    bs = len(samples_list)
    inputs = [input for (input, _) in samples_list]
    input_batch = torch.stack(inputs)

    target_list = [target.ndata['solution'].squeeze(-1) for (_,target) in samples_list]
    target_batch = torch.stack(target_list)
    return (input_batch,target_batch)

def _collate_fn_tensor_edge(samples_list):
    """
    Custom collate function for FGNN dense data with edge embeddings.
    """
    inputs = [input for (input, _) in samples_list]
    input_batch = torch.stack(inputs)

    target_list = [edge_format_to_dense_tensor(target.edata['solution'].squeeze(-1), target) for (_,target) in samples_list]
    target_batch = torch.stack(target_list)
    return (input_batch,target_batch)

def tensor_to_pytorch(generator, embed, batch_size=32, shuffle=False, num_workers=4, **kwargs):
    """
    Prepares the PyTorch Dataloader for dense tensors used by FGNNs.
    """
    if embed=='edge':
        collate_fn = _collate_fn_tensor_edge
    elif embed=='node':
        collate_fn = _collate_fn_tensor_node
    else:
        raise NotImplementedError(f'Collate function for dense tensor and embedding {embed} has not been implemented.')
    pytorch_loader = DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return pytorch_loader

def _collate_fn_mt(samples_list):
    """
    Custom collate function for FGNN dense masked tensor data.
    We assume that we got a list of tensors of dimension 3 with dims 1 and 2 equal, and that all tensors have the same n_features : shape=(n_vertices,n_vertices,n_features)
    """
    input1_list = [input1 for (input1, _) in samples_list]
    target_list = [target for (_,target) in samples_list]
    input_mt = maskedtensor.from_list(input1_list,dims=(0,1))
    target_mt = maskedtensor.from_list(target_list,dims=(0,1))
    return (input_mt,target_mt)

def maskedtensor_to_pytorch(generator, batch_size=32, shuffle=False, **kwargs):
    pytorch_loader = DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=0, collate_fn=_collate_fn_mt)
    return pytorch_loader

def _collate_fn_dgl(samples_list):
    """
    Custom collate function for dgl graphs. (We simply use the dgl.batch method with both inputs and targets)
    """
    input1_list = [input1 for (input1, _) in samples_list]
    target_list = [target for (_,target) in samples_list]
    input_batch = dglbatch(input1_list)
    target_batch = dglbatch(target_list)
    return (input_batch,target_batch)

def dgl_to_pytorch(generator, batch_size, shuffle=False, num_workers=4, **kwargs):
    pytorch_loader = DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, collate_fn=_collate_fn_dgl)
    return pytorch_loader

def get_generator(config:dict, type:str):
    """
    Fetches the right generator class for the problem and generates it.
     - config : the config dictionary
     - type : either 'train','val' or 'test.
    """
    problem_key = config['problem'].lower()
    Generator_Class = get_generator_class(problem_key)
    
    lookup_key = TRAIN_VAL_TEST_LOOKUP[type]

    data_config = config['data'][lookup_key]
    data_config['path_dataset'] = config['data']['path_dataset']
    problem_specific_config = config['data'][lookup_key]['problems'].get(problem_key, {})

    dataset = Generator_Class(type, {**data_config, **problem_specific_config})
    return dataset

def get_dataset(config:dict, type:str, dgl_check=True, mt_check=True, dataloader_args={}):
    """
    Loads the dataset needed.
     - config : the config dictionary
     - type : 'train', 'val' or 'test'.
     - dgl_check : if set to True, will raise an error in the case there is an incompatibility between the model loaded and the dataset. If False, will just log a warning.
     - mt_check : same as dgl_check but with masked tensors
     - dataloader_args : additional arguments to pass to the dataloader
    """
    dataset = get_generator(config, type)

    use_dgl = config['arch']['use_dgl']
    arch_name = config['arch']['name'].lower()
    check_dgl_compatibility(use_dgl, arch_name, dgl_check=dgl_check)
    
    use_maskedtensor = config['data']['use_maskedtensor']
    problem = config['problem']
    if not use_dgl: check_maskedtensor_compatibility(use_maskedtensor, problem, force_check=mt_check)

    dataset.load_dataset(use_dgl=use_dgl)

    loader_config = config['train']
    if use_dgl:
        dataloaded = dgl_to_pytorch(dataset, **loader_config, **dataloader_args)
    elif use_maskedtensor:
        dataloaded = maskedtensor_to_pytorch(dataset,**loader_config, **dataloader_args)
    else:
        embed = config['arch']['embedding']
        embed = EMBED_TYPES.get(embed, embed)
        dataloaded = tensor_to_pytorch(dataset, embed,**loader_config, **dataloader_args)

    return dataloaded

"""Some shortcut functions to easily get data generators and data loaders"""

def get_train_val_generators(config:dict):
    train_gen = get_generator(config, 'train')
    val_gen = get_generator(config, 'val')
    return train_gen, val_gen

def get_test_generator(config:dict):
    test_gen = get_generator(config, 'test')
    return test_gen

def get_train_val_datasets(config:dict, dgl_check=True):
    train_dataset = get_dataset(config, 'train', dgl_check=dgl_check, dataloader_args={'shuffle':True})
    val_dataset = get_dataset(config, 'val', dgl_check=dgl_check)
    return train_dataset, val_dataset

def get_test_dataset(config:dict, dgl_check=True):
    test_dataset = get_dataset(config, 'test', dgl_check=dgl_check)
    return test_dataset
    




