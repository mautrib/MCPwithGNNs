from metrics.preprocess import edgefeat_converter, fulledge_converter, node_converter
from metrics.common import fulledge_compute_f1, edgefeat_compute_f1, node_compute_f1, node_total
from metrics.mcp import fulledge_total as mcp_fulledge_total, edgefeat_total as mcp_edgefeat_total, node_total as mcp_node_total
from models.base_model import GNN_Abstract_Base_Class

EMBED_TYPES = {
    'rsnode': 'node',
    'rsedge': 'edge'
}

MCP_VARIANTS = ('mcp', 'mcphard', 'mcptrue', 'mcptruehard')

def get_trainval_fulledge_metric(problem):
    if problem=='mcp':
        return fulledge_compute_f1
    raise NotImplementedError(f"Train/val metric for fulledge problem {problem} has not been implemented.")

def get_test_fulledge_metric(problem):
    if problem in MCP_VARIANTS:
        return fulledge_compute_f1
    raise NotImplementedError(f"Test metric for fulledge problem {problem} has not been implemented.")

def get_trainval_edgefeat_metric(problem):
    if problem=='mcp':
        return edgefeat_compute_f1
    raise NotImplementedError(f"Train/val metric for edge problem {problem} has not been implemented.")

def get_test_edgefeat_metric(problem):
    if problem in MCP_VARIANTS:
        return edgefeat_compute_f1
    raise NotImplementedError(f"Test metric for edge problem {problem} has not been implemented.")

def get_trainval_node_metric(problem):
    if problem in MCP_VARIANTS:
        return node_compute_f1
    raise NotImplementedError(f"Train/val metric for node problem {problem} has not been implemented.")

def get_test_node_metric(problem):
    if problem in MCP_VARIANTS:
        return node_compute_f1
    raise NotImplementedError(f"Test metric for node problem {problem} has not been implemented.")

def get_trainval_metric(eval, problem):
    if eval=='edge':
        eval_fn = get_trainval_edgefeat_metric(problem)
    elif eval=='fulledge':
        eval_fn = get_trainval_fulledge_metric(problem)
    elif eval=='node':
        eval_fn = get_trainval_node_metric(problem)
    else:
        raise NotImplementedError(f"Eval method {eval} not implemented")
    return eval_fn

def get_test_metric(eval, problem):
    if eval=='edge':
        eval_fn = get_test_edgefeat_metric(problem)
    elif eval=='fulledge':
        eval_fn = get_test_fulledge_metric(problem)
    elif eval=='node':
        eval_fn = get_test_node_metric(problem)
    else:
        raise NotImplementedError(f"Eval method {eval} not implemented")
    return eval_fn

def get_preprocessing(embed, eval, problem):
    if embed=='edge':
        if eval=='edge':
            if problem in (MCP_VARIANTS):
                return edgefeat_converter
            else:
                raise NotImplementedError(f"Preprocessing for {embed=}, {eval=}, {problem=} not implemented")
        elif eval=='fulledge':
            if problem in (MCP_VARIANTS):
                return fulledge_converter
            else:
                raise NotImplementedError(f"Preprocessing for {embed=}, {eval=}, {problem=} not implemented")
        else:
            raise NotImplementedError(f"Unknown eval '{eval}' for embedding type 'edge'.")
    elif embed=='node':
        if eval=='node':
            if problem in (MCP_VARIANTS):
                return node_converter
            else:
                raise NotImplementedError(f"Preprocessing for {embed=}, {eval=}, {problem=} not implemented")
        else:
            raise NotImplementedError(f"Unknown eval '{eval}' for embedding type 'edge'.")
    else:
        raise NotImplementedError(f"Embed {embed} not implemented.")

def get_preprocess_additional_args(problem: str, config: dict):
    return {}

def assemble_metric_function(preprocess_function, eval_function, preprocess_additional_args=None):
    if preprocess_additional_args is None:
        preprocess_additional_args = {}
    def final_function(raw_scores, target, **kwargs):
        l_inferred, l_targets, l_adjacency = preprocess_function(raw_scores, target, **kwargs, **preprocess_additional_args)
        try: #We try to add the list of adjacencies to the eval function
            result = eval_function(l_inferred, l_targets, l_adjacency)
        except TypeError as type_error:
            str_error = " ".join(str(type_error).split(' ')[1:])
            if str_error=="takes 2 positional arguments but 3 were given": #The eval function doesn't handle the adjacencies (OLD functions)
                result = eval_function(l_inferred, l_targets)
            else: #In case it's another error, raise it
                raise type_error
        return result
    return final_function

def setup_trainval_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    embed = config['arch']['embedding']
    embed = EMBED_TYPES.get(embed, embed)
    eval = config['arch']['eval']
    try:
        preprocess_function = get_preprocessing(embed, eval, problem)
        eval_fn = get_trainval_metric(eval, problem)
        preprocess_additional_args = get_preprocess_additional_args(problem, config)
        metric_fn = assemble_metric_function(preprocess_function=preprocess_function, eval_function=eval_fn, preprocess_additional_args=preprocess_additional_args)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"There was a problem with the train_val setup metric. I'll let it go anyways, but additional metrics won't be saved. Error stated is: {ne}")

def setup_test_metric(pl_model: GNN_Abstract_Base_Class, config: dict)-> None:
    problem = config['problem']
    embed = config['arch']['embedding']
    embed = EMBED_TYPES.get(embed, embed)
    eval = config['arch']['eval']

    preprocess_function = get_preprocessing(embed, eval, problem)
    eval_fn = get_test_metric(eval, problem)
    preprocess_additional_args = get_preprocess_additional_args(problem, config)
    metric_fn = assemble_metric_function(preprocess_function=preprocess_function, eval_function=eval_fn, preprocess_additional_args=preprocess_additional_args)
    pl_model.attach_metric_function(metric_fn, start_using_metric=True)

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=False, istest=False) -> None:
    """
    Attaches a metric to the Pytorch Lightning model. This metric can be different in train_val and test cases.
    If the metric in test hasn't been implemented, it will try to use the train_val one.
     - pl_model : Pytorch Lightning model, child of GNN_Abstract_Base_Class
     - config   : Config with all the parameters configured as in the file 'default_config.yaml'
     - soft     : if set to False, will raise an error if the train_val metric hasn't been implemented. If True, will let it pass with a warning
    """
    if istest:
        try:
            setup_test_metric(pl_model, config)
            return None
        except NotImplementedError:
            print('Test metric not found, using train_val metric...')
    setup_trainval_metric(pl_model, config, soft)
