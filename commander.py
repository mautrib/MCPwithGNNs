import yaml
import toolbox.utils as utils
import os

from models import get_pipeline, get_pl_model, get_torch_model, get_optim_args, is_dummy
from models.base_model import GNN_Abstract_Base_Class
from data import get_test_dataset, get_train_val_datasets
from metrics import setup_metric
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
import argparse

import toolbox.wandb_helper as wbh

class CB_val_train(Callback):
    def on_validation_start(self, trainer, pl_module):
        trainer.model.train()


def get_config(filename='default_config.yaml') -> dict:
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_observer(config: dict):
    path = config['observers']['base_dir']
    path = os.path.join(os.getcwd(), path)
    utils.check_dir(path)
    observer = config['observers']['observer']
    if observer=='wandb':
        logger = WandbLogger(project=f"{config['project']}_{config['problem']}", log_model="all", save_dir=path)
        try:
            logger.experiment.config.update(config)
        except AttributeError as ae:
            return None
    else:
        raise NotImplementedError(f"Observer {observer} not implemented.")
    return logger

def load_model(config: dict, path: str, add_metric=True, **add_metric_kwargs) -> GNN_Abstract_Base_Class:
    """
     - config : dict. The configuration dictionary (careful, must correspond to the model trying to be loaded). If set to None, will try to download a model from W&B
     - path : str. The local path of the Pytorch Lightning experiment, or the id of the run if need to be fetched on W&B
     - add_metric: bool. Adds an external metric to the pytorch lightninh module.
     - add_metric_kwargs: Arguments passed to the setup_metric function if activated.
    """
    if is_dummy(config['arch']['name']):
        pl_model = get_pipeline(config)
    else:
        print(f'Loading base model from {path}... ', end = "")
        try:
            PL_Model_Class = get_pl_model(config)
            pl_model = PL_Model_Class.load_from_checkpoint(path, model=get_torch_model(config), optim_args=get_optim_args(config))
        except (FileNotFoundError) as e:
            if config['observers']['observer']=='wandb':
                print(f"Failed at finding model locally with error : {e}. Trying to use W&B.")
                project = f"{config['project']}_{config['problem']}"
                wb_config, path = wbh.download_model(project, path)
                PL_Model_Class = get_pl_model(config)
                pl_model = PL_Model_Class.load_from_checkpoint(path, model=get_torch_model(wb_config), optim_args=get_optim_args(wb_config))
            else:
                raise e
        print('Done.')
    if add_metric:
        setup_metric(pl_model, config, **add_metric_kwargs)
    return pl_model

def get_trainer_config(config: dict, only_test=False) -> dict:
    trainer_config = config['train']
    accelerator_config = utils.get_accelerator_dict(config['device'])
    trainer_config.update(accelerator_config)
    if not only_test:
        early_stopping = EarlyStopping('lr', verbose=True, mode='max', patience=1+config['train']['max_epochs'], divergence_threshold=config['train']['optim_args']['lr_stop'])
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, verbose=True)
        trainer_config['callbacks'] = [early_stopping, checkpoint_callback, CB_val_train()]
    clean_config = utils.restrict_dict_to_function(pl.Trainer.__init__, trainer_config)
    return clean_config

def setup_trainer(config: dict, model: GNN_Abstract_Base_Class, watch=True, only_test=False) -> pl.Trainer:
    trainer_config = get_trainer_config(config, only_test=only_test)
    if config['observers']['use']:
        logger = get_observer(config)
        if logger is None:
            print("Logger did not load. Could mean an error or that we are not in the zero_ranked experiment.")
        else:
            if watch: logger.watch(model)
            trainer_config['logger'] = logger
    trainer = pl.Trainer(**trainer_config)
    return trainer

def train(config: dict)->pl.Trainer:
    if is_dummy(config['arch']['name']):
        print("Dummy architecture, can't train.")
        return None
    if config['train']['anew']:
        pl_model = get_pipeline(config)
        setup_metric(pl_model, config)
    else:
        pl_model = load_model(config, config['train']['start_model'])
    trainer = setup_trainer(config, pl_model)
    train_dataset, val_dataset = get_train_val_datasets(config)
    trainer.fit(pl_model, train_dataset, val_dataset)
    return trainer

def test(config: dict, trainer=None, model=None, dataloaders=None, **kwargs) -> None:
    if dataloaders is None: dataloaders = get_test_dataset(config)
    arg_dict = {'dataloaders': dataloaders,
                'verbose':True
    }
    if trainer is None:
        pl_model = model
        if pl_model is None: pl_model = load_model(config, config['train']['start_model'], add_metric=False)
        trainer = setup_trainer(config, pl_model, **kwargs)
    else:
        arg_dict['ckpt_path'] = 'best'
        pl_model = trainer.model
    setup_metric(pl_model, config, istest=True)
    arg_dict['model'] = pl_model
    trainer.test(**arg_dict)
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Main file for creating experiments.')
    parser.add_argument('command', metavar='c', choices=['train','test'],
                    help='Command to execute : train or test')
    parser.add_argument('--config', default='default_config.yaml', type=str, help='Name of the configuration file.')
    args = parser.parse_args()
    if args.command=='train':
        training=True
        default_test = False
    elif args.command=='test':
        training=False
        default_test=True
    
    config = get_config(args.config)
    config = utils.clean_config(config)
    trainer=None
    if training:
        trainer = train(config)
    if default_test or config['test_enabled']:
        test(config, trainer)

if __name__=="__main__":
    pl.seed_everything(3787, workers=True)
    main()
