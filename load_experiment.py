import logging
import hydra
import wandb
import torch
import numpy as np
import dgl
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from data_loaders import GridNavDataModule
from models.graphVAE import LightningGraphVAE
from util.util import *
from util.logger_callbacks import GraphVAELogger
from run_experiments import get_dataset_dir

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def load_experiment(cfg: DictConfig) -> None:

    logger = logging.getLogger(__name__)
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(cfg.seed)

    test_mode = True if cfg.run_name == "test" else False
    dataset_full_dir, cfg.data.dataset.path = get_dataset_dir(cfg.data.dataset, test_mode)
    data_module = GridNavDataModule(dataset_full_dir,
                                    batch_size=cfg.data.dataset.batch_size,
                                    num_samples=cfg.results.num_stored_samples,
                                    transforms=cfg.data.dataset.transforms,
                                    num_workers=cfg.num_cpus)
    data_module.setup()

    model = LightningGraphVAE.load_from_checkpoint(cfg.load.checkpoint_path, accelerator=cfg.accelerator)
    model.eval()

    api = wandb.Api()
    run = api.run(cfg.load.wandb_run_path)

    artifact = api.artifact(cfg.load.resources.latent_spaceval.artifact_name)
    table = artifact.get(cfg.load.resources.latent_spaceval.filepath)
    label_ids = torch.tensor(table.get_column("Label_ids"))
    R_unique = torch.tensor(table.get_column("R_unique"))
    R_reconstructions = torch.tensor(table.get_column("R:Reconstructions", convert_to="numpy"))
    offenders = label_ids[~R_unique]

    data = [data_module.dataset[o] for o in offenders]
    graphs_o, labels = ([d[0] for d in data], [d[1] for d in data])
    graphs_o = dgl.batch(graphs_o).to(model.device)
    _, _, logits_A_o, logits_Fx_o, _, _ = model.all_model_outputs_pathwise(graphs_o, num_samples=cfg.results.num_variational_samples)

    mode_A_o, mode_Fx_o = model.decoder.param_m((logits_A_o, logits_Fx_o))
    test_o = (check_unique(mode_A_o) | check_unique(mode_Fx_o)).tolist()

    data = [data_module.dataset[o] for o in label_ids]
    graphs, labels = ([d[0] for d in data], [d[1] for d in data])

    batch_size = 8
    processed = 0
    MODE_A = []
    MODE_FX = []
    while processed < len(graphs):
        graphs_batch = dgl.batch(graphs[processed:processed+batch_size]).to(model.device)
        _, _, logits_A, logits_Fx, _, _ = model.all_model_outputs_pathwise(graphs_batch, num_samples=cfg.results.num_variational_samples)
        mode_A, mode_Fx = model.decoder.param_m((logits_A, logits_Fx))
        MODE_A.append(mode_A.cpu().detach())
        MODE_FX.append(mode_Fx.cpu().detach())
        processed += batch_size
    # graphs = dgl.batch(graphs).to(model.device)
    # _, _, logits_A, logits_Fx, _, _ = model.all_model_outputs_pathwise(graphs, num_samples=cfg.results.num_variational_samples)

    # mode_A, mode_Fx = model.decoder.param_m((logits_A, logits_Fx))
    MODE_A = torch.cat(MODE_A, dim=0)
    MODE_FX = torch.cat(MODE_FX, dim=0)
    test = (check_unique(MODE_A) & check_unique(MODE_FX)).tolist()

    logger.info("Done.")


if __name__ == "__main__":
    load_experiment()
