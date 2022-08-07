import logging
import os
import hydra
from pathlib import Path #TODO replace by hydra?
from omegaconf import DictConfig, OmegaConf

from torch import optim #TODO remove with lightning
from torch.utils.tensorboard import SummaryWriter #TODO remove with lightning

from data_loaders import Maze_Dataset #TODO (may) remove with lightning
from util.util import *

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def run_experiment(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    #TODO: can remove cfg.models when migrating to Lightning
    model = hydra.utils.instantiate(cfg.models)
    #model = GraphVAE(configuration=cfg.models.configuration, hyperparameters=cfg.models.hyperparameters) #DEBUG
    optimizer = optim.Adam(model.parameters(), lr=cfg.models.hyperparameters.optimiser.learning_rate) #TODO: remove with lightning

    dataset_full_dir, cfg.datasets.path = get_dataset_dir(cfg.datasets)
    train_data, test_data = get_data(dataset_full_dir, transforms=cfg.datasets.transforms)

    cfg.run_name='gnn_features_sg'
    run_name = get_run_name(cfg.run_name, cfg.models, cfg.datasets.path)
    writer = SummaryWriter('runs/' + run_name)

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model, optimizer, model_state_best_training_elbo, \
    optim_state_best_training_elbo, early_t = fit_model_rw(model, optimizer, train_data, cfg, test_data=test_data,
                                                           latent_eval_freq=cfg.models.metrics.plot_every,
                                                           tensorboard=writer)

    if early_t: run_name = run_name + '_early_t'
    save_file = 'checkpoints/' + run_name + '.pt'
    logger.info(f"Saving to {save_file}")
    save_state(cfg, model, optimizer, save_file, [model_state_best_training_elbo], [optim_state_best_training_elbo])

    writer.close()

    logger.info("Done")

def get_data(nav_dir, transforms=None):
    train_data = Maze_Dataset(
        nav_dir, train=True,
        transform=transforms)

    test_data = Maze_Dataset(
        nav_dir, train=False,
        transform=transforms)

    return train_data, test_data

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
    #data_directory = 'test'
    data_full_dir = datasets_dir + data_directory
    return data_full_dir, data_directory

def get_run_name(tag, model_cfg, dataset_dir):

    latent_dim = model_cfg.configuration.shared_parameters.latent_dim
    batch_size = model_cfg.hyperparameters.optimiser.batch_size
    epochs = model_cfg.hyperparameters.optimiser.epochs
    run_name = f"{tag}_{dataset_dir}_-z={latent_dim}_b={batch_size}-e={epochs}"
    return run_name


if __name__ == "__main__":
    run_experiment()