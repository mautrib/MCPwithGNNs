# A pipeline for Graph Neural Networks Training and Testing

In this directory, we provide a pipeline to train and test GNNs on multiple problems.

## Dependencies

The packages we had installed are listed in the `environment.yml`. Feel free to try to use different versions of the packages depending on your needs and updates. To install them, use conda : 
```
conda env create -f environment.yaml
```
To use exact Maximum Clique Problem solvers, install [PMC](https://github.com/ryanrossi/pmc/). It may be needed to `make` the `libpmc.so` file. For that use explicitly the command: 
```
make libpmc.so
```
And then move it to the library directory (`/usr/lib/` on Unix).

## Training and testing basics

To train a model, simply modify the parameters of our pipeline in the [default config file](default_config.yaml), and then run:
```
python commander.py train
```

To only test a model, use:
```
python commander.py test
```

## Implementing your own

### Problem :

Add a generator for your data inheriting from the [Base_Generator](https://github.com/MauTrib/gnn-en-folie/blob/main/data/base.py#L6) class, with an implemented  `self.compute_example` function, computing a dense tensor representation and a `self._solution_conversion` function, converting the target to a DGL format.

### Model : 

Implement it in the `models` directory, using pytorch. Then, add it to the [`MODULE_DICT`](https://github.com/MauTrib/gnn-en-folie/blob/main/models/__init__.py#L19) variable, with its returning embedding (`edge` or `node`). If it uses dense tensors (like FGNNs), add its name to the [`NOT_DGL_ARCHS`](https://github.com/MauTrib/gnn-en-folie/blob/main/models/__init__.py#L25) variable. It should then automatically use the corresponding Pytorch Lightning wrapper.

### Metric :

There are for now two modes of evaluating metrics:
 - `edge` : using, like DGL, only the edges present in the original adjacency graph
 - `fulledge` : using all the edges, as in the form of an adjacency matrix

You can add your own metric using these formats and add it to the corresponding `get` function in the [`metrics.__init__.py`](https://github.com/MauTrib/gnn-en-folie/blob/main/metrics/__init__.py) file.

Values of your metric should be returned as a dictionary.

## Project Structure

```bash
.
├── data # Generators
|   └── __init__.py # Handles the data selection and generation
|   └── base.py # Base abstract class of generators
|   └── graph_generation.py.py # Graph generation tools
|   └── mcp.py # Maximum Clique Problem
├── metrics
|   └── __init__.py # Handles the metrics selection
|   └── common.py # Metrics used in multiple problems
|   └── mcp.py
|   └── preprocess.py # Tool for handling DGL and dense tensor conversions
├── models # Implementation of models
|   └── dgl
|       └── gat.py
|       └── gatedgcn.py
|       └── gcn.py
|       └── gin.py
|   └── fgnn
|       └── fgnn.py
|       └── layers.py # Base FGNN layers
|   └── __init__.py # Handles the model selection
|   └── dgl_edge.py # Model using DGL returning edge embeddings
|   └── dgl_node.py # Model using DGL returning node embeddings
|   └── fgnn_edge.py # FGNN returning edge embeddings
|   └── fgnn_node.py # FGNN returning node embeddings
├── toolbox
|   └── searches # Folder with the post-treatment algorithms
|       └── mcp_beamsearch.py #Post-inference data processing to generate real cliques
|       └── mcp_solver.py #Parallel solving of MCP
|   └── conversions.py # Useful conversions between Dense and DGL formats
|   └── maskedtensor.py # Implementation of Masked Tensors
|   └── planner.py # Planning and checkpointing system
|   └── utils.py
|   └── wandb_helper.py # Useful interface functions for W&B
├── commander.py # Main entry of the pipeline
├── default_config.yaml # Configuration file

```



