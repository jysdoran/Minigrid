import logging
import os
import sys
import hydra
from pathlib import Path #TODO replace by hydra?
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl

from data_loaders import GridNavDataModule
from models.graphVAE import GraphVAE, LightningGraphVAE
from util.util import *

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def run_experiment(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Working directory : {}".format(os.getcwd()))
    process_cfg(cfg)
    seed_everything(cfg.seed)

    dataset_full_dir, cfg.datasets.path = get_dataset_dir(cfg.datasets)
    data_module = GridNavDataModule(dataset_full_dir, batch_size=cfg.datasets.batch_size,  transforms=cfg.datasets.transforms)

    model = hydra.utils.instantiate(config=cfg.models,
                                    config_model=cfg.models.configuration,
                                    config_optim=cfg.optim,
                                    hparams_model=cfg.models.hyperparameters,
                                    _recursive_=False)
    # model_lightning = LightningGraphVAE(config_model=cfg.models.configuration, config_optim=cfg.optim,
    #                                     hparams_model=cfg.models.hyperparameters)# DEBUG

    #writer = SummaryWriter('.')

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(accelerator=cfg.accelerator, devices=cfg.num_devices, max_epochs=cfg.epochs)
    trainer.fit(model, data_module)

    # model, optimizer, model_state_best_training_elbo, \
    # optim_state_best_training_elbo, early_t = fit_model_rw(model, optimizer, train_data, cfg, test_data=test_data,
    #                                                        latent_eval_freq=cfg.results.plot_every,
    #                                                        tensorboard=writer)

    # model_checkpoint_filename = cfg.run_name #get_run_name(cfg.run_name, cfg.models, cfg.datasets.path)
    # if early_t: model_checkpoint_filename = model_checkpoint_filename + '_early_t'
    # save_file = model_checkpoint_filename + '.pt'
    # logger.info(f"Saving to {save_file}")
    # save_state(cfg, model, optimizer, save_file, [model_state_best_training_elbo], [optim_state_best_training_elbo])

    # writer.close()
    logger.info("Done")

def get_dataset_dir(cfg):
    base_dir = str(Path(__file__).resolve().parent)
    datasets_dir = base_dir + '/datasets/'

    dataset_size = cfg.size
    task_structures = cfg.task_structures  # ('rooms_unstructured_layout','maze')
    data_type = cfg.data_type  # 'graph'
    data_dims = cfg.gridworld_data_dim
    data_dim = data_dims[-1]
    task_structures = '-'.join(task_structures)

    attributes_dim = len(cfg.node_attributes)
    encoding = cfg.encoding

    data_directory = f"ts={task_structures}-x={data_type}-s={dataset_size}-d={data_dim}-f={attributes_dim}-enc={encoding}"
    data_directory = 'test'
    data_full_dir = datasets_dir + data_directory
    return data_full_dir, data_directory

def process_cfg(cfg):
    # sets all of the Auto arguments

    if cfg.datasets.data_type =='graph':
        if cfg.datasets.encoding =='minimal':
            gw_data_dim = cfg.datasets.gridworld_data_dim
            f = lambda x : (x - 1) / 2
            cfg.datasets.max_nodes = int(f(gw_data_dim[1]) * f(gw_data_dim[2]))
            cfg.models.configuration.decoder.output_dim.adjacency = int((cfg.datasets.max_nodes - 1)*2)

# #obsolete
# def get_run_name(tag, model_cfg, dataset_dir):
#
#     latent_dim = model_cfg.configuration.shared_parameters.latent_dim
#     batch_size = model_cfg.hyperparameters.optimiser.batch_size
#     epochs = model_cfg.hyperparameters.optimiser.epochs
#     run_name = f"{tag}_{dataset_dir}_-z={latent_dim}_b={batch_size}-e={epochs}"
#     return run_name

# def get_data(nav_dir, transforms=None):
#     train_data = GridNav_Dataset(
#         nav_dir, train=True,
#         transform=transforms)
#
#     test_data = GridNav_Dataset(
#         nav_dir, train=False,
#         transform=transforms)
#
#     return train_data, test_data


if __name__ == "__main__":
    sys.argv.append('run_name=gnn_features_sg')
    run_experiment()