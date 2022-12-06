import copy
import logging

import dgl
import networkx as nx
import numpy as np
import torchvision
from typing import List, Union, Tuple, Optional, Any, Dict
from collections import defaultdict
import torch
import pytorch_lightning as pl
import wandb
import einops
import pandas as pd
from PIL import Image as PILImage
from envs.multigrid.multigrid import Grid

from . import transforms as tr
from .graph_metrics import compute_metrics
from .util import check_unique, cdist_polar, lerp, slerp, rgba2rgb
from copy import deepcopy

logger = logging.getLogger(__name__)

class GraphVAELogger(pl.Callback):
    def __init__(self,
                 label_contents: Dict[str, Any],
                 samples: Dict[str, Tuple[dgl.DGLGraph, torch.Tensor]],
                 logging_cfg,
                 dataset_cfg,
                 label_descriptors_config: Dict = None,
                 accelerator: str = "cpu"):

        super().__init__()
        device = torch.device("cuda" if accelerator == "gpu" else "cpu")
        self.logging_cfg = logging_cfg
        self.dataset_cfg = dataset_cfg

        self.label_contents = label_contents
        self.label_descriptors_config = label_descriptors_config
        self.force_valid_reconstructions = logging_cfg.force_valid_reconstructions
        self.num_stored_samples = logging_cfg.num_stored_samples
        self.num_image_samples = logging_cfg.num_image_samples
        self.num_embedding_samples = logging_cfg.num_embedding_samples
        self.num_generated_samples = logging_cfg.num_generated_samples
        self.num_variational_samples_logging = logging_cfg.num_variational_samples
        self.max_cached_batches = self.num_stored_samples // self.dataset_cfg.batch_size
        self.max_cached_batches = max(self.max_cached_batches, 1) #store at least one batch
        self.attributes = dataset_cfg.node_attributes
        self.graphs = {}
        self.labels = {}
        self.gw = {}
        self.imgs = {}
        self.node_to_gw_mapping = tr.DilationTransform(1)
        self.batches_prepared = False
        self.batch_template = {
            "graphs": [],
            "label_ids": [],
            }
        self.outputs_template = {key:[] for key in self.logging_cfg.model_outputs}

        self.validation_batch = deepcopy(self.batch_template)
        self.predict_batch = deepcopy(self.batch_template)
        self.validation_step_outputs = deepcopy(self.outputs_template)
        self.predict_step_outputs = deepcopy(self.outputs_template)

        try:
            for key in ["train", "val", "test"]:
                self.graphs[key], self.labels[key] = samples[key]
                self.graphs[key] = dgl.unbatch(self.graphs[key])
                self.graphs[key] = self.graphs[key]
                self.labels[key] = self.labels[key]
                self.graphs[key] = dgl.batch(self.graphs[key]).to(device)
                self.labels[key] = self.labels[key].to(device)
                if len(self.labels[key]) < self.num_stored_samples:
                    raise ValueError(f"Not enough samples for {key} data module."
                                     f"Number of samples available for storage in data module: {len(self.labels[key])}."
                                     f"Number of samples to be stored: {self.num_stored_samples}.")
        except KeyError as e: #TODO: why does it catch the value error?
            logger.info(f"{key} dataset was not supplied in the samples provided to the GraphVAELogger")

        for key in self.graphs.keys():
            if self.dataset_cfg.encoding == "minimal": #TODO: uniformise with new standard
                self.gw[key] = self.encode_graph_to_gridworld(self.graphs[key], self.attributes).to(device)
                self.imgs[key] = self.gridworld_to_img(self.gw[key][:self.num_image_samples]).to(device)
            elif self.dataset_cfg.encoding == "dense":
                if "images" in self.label_contents.keys():
                    self.gw[key] = None
                    self.imgs[key] = self.label_contents["images"][self.labels[key]][:self.num_image_samples]
                else:
                    raise RuntimeError("Images not found in label contents. Cannot log images.")

        self.resize_transform = torchvision.transforms.Resize((100, 100),
                                               interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    # Main logging logic

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        try:
            self.log_images(trainer, "dataset/train", self.imgs["train"], self.labels["train"], mode="RGB")
            self.log_images(trainer, "dataset/val", self.imgs["val"], self.labels["val"], mode="RGB")
            self.log_images(trainer, "dataset/test", self.imgs["test"], self.labels["test"], mode="RGB")
        except KeyError as e:
            logger.info(f"{e} dataset was not supplied in the samples provided to the GraphVAELogger")

    def on_validation_epoch_end(self, trainer, pl_module):
        _ = GraphVAELogger.prepare_stored_batch(self.predict_batch, self.predict_step_outputs, self.num_stored_samples)
        self.batches_prepared = GraphVAELogger.prepare_stored_batch(self.validation_batch, self.validation_step_outputs, self.num_stored_samples)
        self.log_epoch_metrics(trainer, pl_module, self.predict_step_outputs, "predict")
        self.log_epoch_metrics(trainer, pl_module, self.validation_step_outputs, "val")
        if "train" in self.graphs.keys():
            outputs = self.obtain_model_outputs(self.graphs["train"], pl_module, num_samples=self.num_image_samples, num_var_samples=1)
            reconstructed_imgs_train = self.obtain_imgs(outputs, pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["train"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/train", reconstructed_imgs_train, captions=captions, mode="RGB")
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/train", outputs=outputs)

            if self.num_variational_samples_logging > 0:
                outputs = self.obtain_model_outputs(self.graphs["train"], pl_module, num_samples=self.num_image_samples, num_var_samples=self.num_variational_samples_logging)
                self.log_epoch_metrics(trainer, pl_module, outputs, f"predict_{self.num_variational_samples_logging}_var_samples")
                reconstructed_imgs_train = self.obtain_imgs(outputs, pl_module)
                captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["train"], outputs["unweighted_elbos"])]
                self.log_images(trainer, f"reconstructions/train/{self.num_variational_samples_logging}_var_sample", reconstructed_imgs_train, captions=captions, mode="RGB")
                self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module,
                                       tag=f"reconstructions/train/{self.num_variational_samples_logging}_var_sample",
                                       outputs=outputs)

        if "val" in self.graphs.keys():
            outputs = self.obtain_model_outputs(self.graphs["val"], pl_module, num_samples=self.num_image_samples, num_var_samples=1)

            reconstructed_imgs_val = self.obtain_imgs(outputs, pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["val"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/val", reconstructed_imgs_val, captions=captions, mode="RGB")
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/val",  outputs=outputs)

        self.log_prior_sampling(trainer, pl_module, tag=None)

    def log_epoch_metrics(self, trainer, pl_module, outputs, mode):

        if not isinstance(outputs, dict):
            raise NotImplementedError("GraphVAElogger.log_epoch_metrics() - only handles outputs in dict format")

        to_log = {
            f"metric/mean/std/{mode}"   : torch.linalg.norm(outputs["mean"], dim=-1).std().item(),
            f"metric/sigma/mean/{mode}" : torch.square(outputs["std"]).sum(axis=-1).sqrt().mean().item() / outputs["std"].shape[-1],
            f'metric/entropy/Fx/{mode}' : pl_module.decoder.entropy_A(outputs["logits_Fx"]).sum(),
            }
        for key in outputs.keys():
            to_log[f"metric/{key}/{mode}"] = outputs[key].mean()

        trainer.logger.log_metrics(to_log, step=trainer.global_step)

        flattened_logits_Fx = torch.flatten(pl_module.decoder.param_pFx(outputs["logits_Fx"]))
        trainer.logger.experiment.log(
            {f"distributions/logits/Fx/{mode}": wandb.Histogram(flattened_logits_Fx.to("cpu")),
             "global_step": trainer.global_step})

        if "logits_A" in outputs.keys():
            to_log[f'metric/entropy/A/{mode}'] = pl_module.decoder.entropy_A(outputs["logits_A"]).sum()
            flattened_logits_A = torch.flatten(pl_module.decoder.param_pA(outputs["logits_A"]))
            trainer.logger.experiment.log(
                {f"distributions/logits/A/{mode}": wandb.Histogram(flattened_logits_A.to("cpu")),
                 "global_step": trainer.global_step})



    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Entering on_train_end()")
        self.log_latent_embeddings(trainer, pl_module, "latent_space", mode="val")

        if self.num_variational_samples_logging > 0:
            outputs = self.obtain_model_outputs(self.graphs["val"], pl_module, num_samples=self.num_embedding_samples, num_var_samples=self.num_variational_samples_logging)
            self.log_latent_embeddings(trainer, pl_module, f"latent_space/{self.num_variational_samples_logging}_var_samples/val",
                                       mode="custom", outputs=outputs)

        self.log_latent_interpolation(trainer, pl_module, tag="interpolation/polar",
                                      mode="val",
                                      num_samples=self.logging_cfg.sample_interpolation.num_samples,
                                      interpolations_per_samples=self.logging_cfg.sample_interpolation.num_interpolations,
                                      remove_dims=self.logging_cfg.sample_interpolation.remove_noninformative_dimensions,
                                      dim_reduction_threshold=self.logging_cfg.sample_interpolation.dimension_reduction_threshold,
                                      latent_sampling=self.logging_cfg.sample_interpolation.sample_latents,
                                      interpolation_scheme=self.logging_cfg.sample_interpolation.interpolation_scheme)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Entering on_predict_end()")
        self.batches_prepared = GraphVAELogger.prepare_stored_batch(self.predict_batch, self.predict_step_outputs, self.num_stored_samples)
        self.log_latent_embeddings(trainer, pl_module, "latent_space", mode="prior")
        self.log_latent_embeddings(trainer, pl_module, "latent_space", mode="predict")
        if self.num_variational_samples_logging > 0:
            outputs = self.obtain_model_outputs(self.graphs["train"], pl_module, num_samples=self.num_embedding_samples, num_var_samples=self.num_variational_samples_logging)
            self.log_latent_embeddings(trainer, pl_module, f"latent_space/{self.num_variational_samples_logging}_var_samples/train",
                                       mode="custom", outputs=outputs)

        self.log_latent_interpolation(trainer, pl_module, "interpolation/polar",
                                      mode="predict",
                                      num_samples=self.logging_cfg.sample_interpolation.num_samples,
                                      interpolations_per_samples=self.logging_cfg.sample_interpolation.num_interpolations,
                                      remove_dims=self.logging_cfg.sample_interpolation.remove_noninformative_dimensions,
                                      dim_reduction_threshold=self.logging_cfg.sample_interpolation.dimension_reduction_threshold,
                                      latent_sampling=self.logging_cfg.sample_interpolation.sample_latents,
                                      interpolation_scheme=self.logging_cfg.sample_interpolation.interpolation_scheme)
        self.log_latent_interpolation(trainer, pl_module, "interpolation/polar",
                                      mode="prior",
                                      num_samples=self.logging_cfg.sample_interpolation.num_samples,
                                      interpolations_per_samples=self.logging_cfg.sample_interpolation.num_interpolations,
                                      remove_dims=self.logging_cfg.sample_interpolation.remove_noninformative_dimensions,
                                      dim_reduction_threshold=self.logging_cfg.sample_interpolation.dimension_reduction_threshold,
                                      latent_sampling=self.logging_cfg.sample_interpolation.sample_latents,
                                      interpolation_scheme=self.logging_cfg.sample_interpolation.interpolation_scheme)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Entering on_test_end()")
        if "test" in self.graphs.keys():
            outputs = self.obtain_model_outputs(self.graphs["test"], pl_module,
                                                num_samples=self.num_image_samples, num_var_samples=1)
            reconstructed_imgs = self.obtain_imgs(outputs, pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["test"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/test", reconstructed_imgs, captions=captions, mode="RGB")

    # Boiler plate code

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Starting new validation epoch...")
        self.clear_stored_batches()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Starting new train epoch...")
        self.clear_stored_batches()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Starting new predict epoch...")
        self.clear_stored_batches()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Starting new test epoch...")
        self.clear_stored_batches()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Tuple,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx < self.max_cached_batches:
            if dataloader_idx == 0:
                logger.debug(f"Storing validation batch {batch_idx}.")
                self.store_batch(self.validation_batch, self.validation_step_outputs, batch, outputs)
            else:
                logger.debug(f"Storing predict batch {batch_idx}.")
                self.store_batch(self.predict_batch, self.predict_step_outputs, batch, outputs)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Tuple,
        batch: Any,
        batch_idx: int,
        data_loader_idx: int = 0,
    ) -> None:
        logger.debug(f"Processed predict batch {batch_idx}.")
        if batch_idx < self.max_cached_batches:
            logger.debug(f"Storing predict batch {batch_idx} out of {self.max_cached_batches}.")
            self.store_batch(self.predict_batch, self.predict_step_outputs, batch, outputs)

    # Logging logic

    def log_latent_embeddings(self, trainer, pl_module, tag, mode="val", outputs=None, interpolation=False):

        if self.force_valid_reconstructions:
            raise NotImplementedError("force_valid_reconstructions not implemented yet.")

        if interpolation:
            input_batch = {}
            assert outputs is not None, "Setting the interpolation flag requires outputs to be passed as well."
            outputs = copy.copy(outputs)
            assert outputs["Z"] is not None and len(outputs.keys()) == 1, \
                "Setting the interpolation flag restricts outputs to contain only Z as key."
            outputs["logits_Fx"] = pl_module.decoder(outputs["Z"])
        else:
            if mode == "prior":
                input_batch = {}
                assert outputs is None, "mode=prior with interpolation=false does not support passing outputs in."
                outputs = {}
                Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
                outputs["Z"] = torch.randn(self.num_generated_samples, Z_dim).to(device=pl_module.device)
                outputs["logits_Fx"] = pl_module.decoder(outputs["Z"])
            elif mode == "val":
                input_batch = copy.copy(self.validation_batch)
                outputs = copy.copy(self.validation_step_outputs)
            elif mode == "predict":
                input_batch = copy.copy(self.predict_batch)
                outputs = copy.copy(self.predict_step_outputs)
            elif mode == "custom":
                input_batch = {}
                assert outputs is not None, "Setting the custom flag requires outputs to be passed as well."
                outputs = copy.copy(outputs)
            else:
                raise ValueError(f"Mode {mode} not supported.")

            if mode != "prior":
                for key in outputs.keys():
                    outputs[key] = outputs[key][:self.num_embedding_samples]
                for key in input_batch.keys():
                    input_batch[key] = input_batch[key][:self.num_embedding_samples]
                outputs["Z"] = outputs["mean"]

        if input_batch.get("graphs") is not None:
            input_imgs = tr.Nav2DTransforms.dense_graph_to_minigrid_render(input_batch["graphs"],
                                                                     tile_size=self.logging_cfg.tile_size)

        assert outputs["Z"].shape[0] == \
               outputs["logits_Fx"].shape[0], \
            f"log_latent_embeddings(): Shape mismatch Z={outputs['Z'].shape}, logits_Fx={outputs['logits_Fx'].shape}"

        reconstructed_graphs = pl_module.decoder.to_graph(outputs["logits_Fx"], probabilistic_graph=True, make_batch=False)
        reconstructed_imgs = self.obtain_imgs(outputs, pl_module)
        reconstruction_metrics = {k: [] for k in ["valid","solvable","shortest_path", "resistance", "navigable_nodes"]}
        mode_Fx = pl_module.decoder.param_m(outputs["logits_Fx"])
        start_nodes_ids = mode_Fx[..., pl_module.decoder.attributes.index('start')].argmax(dim=-1)
        goal_nodes_ids = mode_Fx[..., pl_module.decoder.attributes.index('goal')].argmax(dim=-1)
        is_valid = tr.Nav2DTransforms.check_validity_start_goal_dense(
            start_nodes_ids, goal_nodes_ids, mode_Fx[..., pl_module.decoder.attributes.index('active')])
        reconstruction_metrics["valid"] = is_valid.tolist()
        reconstruction_metrics, rec_graphs_nx = \
            compute_metrics(reconstructed_graphs, reconstruction_metrics, start_nodes_ids, goal_nodes_ids,
                            labels=input_batch.get("label_ids"))
        reconstruction_metrics["unique"] = check_unique(mode_Fx).tolist()

        del outputs["logits_Fx"], start_nodes_ids, goal_nodes_ids, rec_graphs_nx, is_valid, reconstructed_graphs

        if self.label_descriptors_config is not None:
            reconstruction_metrics = self.normalise_metrics(reconstruction_metrics, device=pl_module.device)
            logger.info(f"log_latent_embeddings(): graph metrics normalised.")

        # Log average metrics
        to_log = {}
        for key in reconstruction_metrics.keys():
            data = torch.tensor(reconstruction_metrics[key]).to(pl_module.device, torch.float)
            data = data[~torch.isnan(data)]
            to_log[f'metric/{tag}/task_metric/R/{key}/{mode}'] = data.mean()
        if outputs.get("unweighted_elbos") is not None:
            to_log[f'metric/unweighted_elbo/{mode}'] = outputs["unweighted_elbos"].mean()
        trainer.logger.log_metrics(to_log, step=trainer.global_step)
        del to_log
        logger.info(f"log_latent_embeddings(): logged metrics to wandb.")

        # Create embeddings table
        df = pd.DataFrame()
        df.insert(0, "Z", outputs["Z"].tolist())

        # Store pre-computed Input metrics in the embeddings table
        if input_batch.get("label_ids") is not None:
            for input_property in reversed(["shortest_path", "resistance", "navigable_nodes", "task_structure", "seed"]):
                if isinstance(self.label_contents[input_property], list):
                    data = [self.label_contents[input_property][i] for i in input_batch["label_ids"]]
                else:
                    data = self.label_contents[input_property][input_batch["label_ids"]]
                df.insert(0, f"I_{input_property}", data)


        # Store Reconstruction metrics in the embeddings table
        for key in reversed(["valid", "solvable", "unique", "shortest_path", "resistance", "navigable_nodes"]):
            df.insert(0, f"R_{key}", reconstruction_metrics[key])
            if key in ["shortest_path", "resistance", "navigable_nodes"]:
                hist_data = torch.tensor(reconstruction_metrics[key]).to(pl_module.device, torch.float)
                hist_data = hist_data[~hist_data.isnan()].cpu().numpy()
                trainer.logger.experiment.log(
                    {
                        f"distributions/{tag}/R/{key}/{mode}": wandb.Histogram(hist_data),
                        "global_step"                   : trainer.global_step
                        })
        del reconstruction_metrics, hist_data
        logger.info(f"log_latent_embeddings(): logged graph metrics distributions to wandb.")
        if outputs.get("unweighted_elbos") is not None:
            df.insert(0, "Unweighted_elbos", outputs["unweighted_elbos"].tolist())
        df.insert(0, "R:Reconstructions", [wandb.Image(x, mode="RGB") for x in reconstructed_imgs])
        if input_batch.get("label_ids") is not None:
            df.insert(0, "I:Inputs", [wandb.Image(x, mode="RGB") for x in input_imgs])
            df.insert(0, "Label_ids", input_batch["label_ids"].tolist())


        trainer.logger.experiment.log({
            f"tables/{tag}/{mode}": wandb.Table(
                dataframe=df
            ),
            "global_step": trainer.global_step
        })
        logger.info(f"log_latent_embeddings(): Logged dataframe tables/{tag}/{mode} with {len(reconstructed_imgs)} embeddings to wandb.")


    def _log_latent_embeddings_old(self, trainer, pl_module, tag, mode="val", outputs=None):

        logger.info(f"Progression: Entering log_latent_embeddings(), mode:{mode}")

        graphs, labels, logits_A, logits_Fx = [None]*4

        if mode == "prior" or "interpolation" in tag:
            try:
                assert outputs is not None
                Z = outputs["Z"]
            except (KeyError, AssertionError) as e:
                if "interpolation" in tag:
                    raise e("log_latent_embeddings() - Latent interpolation requires outputs['Z'] to be specified.")
                else:
                    Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
                    Z = torch.randn(self.num_generated_samples, Z_dim).to(device=pl_module.device)
            logits_A, logits_Fx = pl_module.decoder(Z)
            unweighted_elbos = None
            labels = None
            if pl_module.predictor is not None:
                y_hat = pl_module.predictor(Z)
                reconstructed_graphs, start_nodes, goal_nodes, is_valid = \
                    tr.Nav2DTransforms.encode_decoder_output_to_graph(logits_A, logits_Fx, pl_module.decoder,
                                                                      correct_A=True)
                y = pl_module.predictor.target_metric_fn(reconstructed_graphs, start_nodes, goal_nodes).to(Z.device)
                y = einops.repeat(y, 'b -> b 1') # (B,) -> (B,1)
                predictor_loss_fn = pl_module.predictor.loss_fn(reduction="none")
                predictor_loss_unreg = predictor_loss_fn(y_hat, y)
            else:
                y_hat = None
                y = None
                predictor_loss_unreg = None
        elif mode == "val":
            Z = self.validation_step_outputs["mean"][:self.num_embedding_samples]
            logits_A, logits_Fx = self.validation_step_outputs["logits_A"][:self.num_embedding_samples], self.validation_step_outputs["logits_Fx"][:self.num_embedding_samples]
            graphs = self.validation_batch["graphs"][:self.num_embedding_samples]
            labels = self.validation_batch["label_ids"][:self.num_embedding_samples]
            unweighted_elbos = self.validation_step_outputs["unweighted_elbos"][:self.num_embedding_samples]
            predictor_loss_unreg = self.validation_step_outputs["predictor_loss_unreg"][:self.num_embedding_samples]
            y_hat = self.validation_step_outputs["y_hat"][:self.num_embedding_samples]
            y = self.validation_step_outputs["y"][:self.num_embedding_samples]
        elif mode == "predict":
            Z = self.predict_step_outputs["mean"][:self.num_embedding_samples]
            logits_A, logits_Fx = self.predict_step_outputs["logits_A"][:self.num_embedding_samples], self.predict_step_outputs["logits_Fx"][:self.num_embedding_samples]
            graphs = self.predict_batch["graphs"][:self.num_embedding_samples]
            labels = self.predict_batch["label_ids"][:self.num_embedding_samples]
            unweighted_elbos = self.predict_step_outputs["unweighted_elbos"][:self.num_embedding_samples]
            predictor_loss_unreg = self.predict_step_outputs["predictor_loss_unreg"][:self.num_embedding_samples]
            y_hat = self.predict_step_outputs["y_hat"][:self.num_embedding_samples]
            y = self.predict_step_outputs["y"][:self.num_embedding_samples]
        elif mode == "custom":
            unweighted_elbos = outputs["unweighted_elbos"]
            logits_A = outputs["logits_A"]
            logits_Fx = outputs["logits_Fx"]
            Z = outputs["mean"]
            y_hat = outputs["y_hat"]
            y = outputs["y"]
            predictor_loss_unreg = outputs["predictor_loss_unreg"]
        else:
            raise RuntimeError(f"log_latent_embeddings() - Invalid mode: {mode}")


        assert Z.shape[0] == logits_A.shape[0] == logits_Fx.shape[0], f"log_latent_embeddings(): Shape mismatch Z={Z.shape}, logits_A={logits_A.shape}, logits_Fx={logits_Fx.shape}"

        if graphs is not None:
            input_gws = self.encode_graph_to_gridworld(graphs, self.attributes)
            if not check_unique(input_gws).all():
                logger.warning(f"Some of the sampled graphs in the {mode} dataset are not unique!")
            input_imgs = self.gridworld_to_img(input_gws)
            del input_gws, graphs

        reconstructed_graphs, start_nodes, goal_nodes, is_valid = \
            tr.Nav2DTransforms.encode_decoder_output_to_graph(logits_A, logits_Fx, pl_module.decoder, correct_A=True)

        reconstruction_metrics = {k: [] for k in ["valid","solvable","shortest_path", "resistance", "navigable_nodes"]}
        reconstruction_metrics["valid"] = is_valid.tolist()
        reconstruction_metrics, rec_graphs_nx = \
            compute_metrics(reconstructed_graphs, reconstruction_metrics, start_nodes, goal_nodes, labels=labels)
        mode_A, mode_Fx = pl_module.decoder.param_m((logits_A, logits_Fx))
        reconstruction_metrics["unique"] = (check_unique(mode_A) | check_unique(mode_Fx)).tolist()
        del is_valid, reconstructed_graphs

        reconstructed_imgs = self.obtain_imgs(outputs, pl_module)

        if self.force_valid_reconstructions:
            probs_Fx = pl_module.decoder.param_pFx(logits_Fx)
            mode_Fx[..., pl_module.decoder.attributes.index("start")],\
            mode_Fx[..., pl_module.decoder.attributes.index("goal")], \
            start_nodes_valid, goal_nodes_valid, \
                is_valid = \
                tr.Nav2DTransforms.force_valid_layout(
                rec_graphs_nx,
                probs_Fx[..., pl_module.decoder.attributes.index("start")],
                probs_Fx[..., pl_module.decoder.attributes.index("goal")])

            reconstruction_metrics_force_valid = {k: [] for k in ["valid","solvable","shortest_path", "resistance", "navigable_nodes"]}
            reconstruction_metrics_force_valid["valid"] = is_valid
            reconstruction_metrics_force_valid, _, = compute_metrics(rec_graphs_nx, reconstruction_metrics_force_valid,
                                                                     start_nodes_valid, goal_nodes_valid, labels=labels)
            mode_A = pl_module.decoder.param_mA(logits_A)
            reconstruction_metrics_force_valid["unique"] = (check_unique(mode_A) | check_unique(mode_Fx)).tolist()
            del start_nodes_valid, goal_nodes_valid
            logger.info(f"log_latent_embeddings(): successfully obtained metrics for the force valid decoded graphs.")
            reconstructed_imgs_force_valid = self.obtain_imgs({"logits_A":logits_A, "logits_Fx": mode_Fx}, pl_module)
            del mode_A, mode_Fx

        del logits_A, logits_Fx, start_nodes, goal_nodes, rec_graphs_nx

        if self.label_descriptors_config is not None:
            reconstruction_metrics = self.normalise_metrics(reconstruction_metrics, device=pl_module.device)
            if self.force_valid_reconstructions:
                reconstruction_metrics_force_valid = self.normalise_metrics(reconstruction_metrics_force_valid, device=pl_module.device)
            logger.info(f"log_latent_embeddings(): graph metrics normalised.")

        to_log = {}
        for key in reconstruction_metrics.keys():
            data = torch.tensor(reconstruction_metrics[key]).to(pl_module.device, torch.float)
            data = data[~torch.isnan(data)]
            to_log[f'metric/{tag}/task_metric/R/{key}/{mode}'] = data.mean()
        if self.force_valid_reconstructions:
            for key in ["valid", "unique", "shortest_path", "resistance", "navigable_nodes"]:
                data = torch.tensor(reconstruction_metrics_force_valid[key]).to(pl_module.device, torch.float)
                data = data[~torch.isnan(data)]
                to_log[f'metric/{tag}/task_metric/RV/{key}/{mode}'] = data.mean()
        if unweighted_elbos is not None:
            to_log[f'metric/unweighted_elbo/{mode}'] = unweighted_elbos.mean()
        trainer.logger.log_metrics(to_log, step=trainer.global_step)
        del to_log
        logger.info(f"log_latent_embeddings(): logged metrics to wandb.")

        df = pd.DataFrame()
        df.insert(0, "Z", Z.tolist())
        if labels is not None:
            # Store pre-computed Input metrics
            for input_property in reversed(["shortest_path", "resistance", "navigable_nodes", "task_structure", "seed"]):
                if isinstance(self.label_contents[input_property], list):
                    data = [self.label_contents[input_property][i] for i in labels]
                else:
                    data = self.label_contents[input_property][labels]
                df.insert(0, f"I_{input_property}", data)

        # Store Force Valid Reconstruction metrics
        if self.force_valid_reconstructions:
            for key in reversed(["valid", "unique", "shortest_path", "resistance", "navigable_nodes"]):
                df.insert(0, f"RV_{key}", reconstruction_metrics_force_valid[key])
                if key in ["shortest_path", "resistance", "navigable_nodes"]:
                    hist_data = torch.tensor(reconstruction_metrics_force_valid[key]).to(pl_module.device, torch.float)
                    hist_data = hist_data[~hist_data.isnan()].cpu().numpy()
                    trainer.logger.experiment.log(
                        {
                            f"distributions/{tag}/RV/{key}/{mode}": wandb.Histogram(hist_data),
                            "global_step" : trainer.global_step
                            })
            del reconstruction_metrics_force_valid

        # Store Reconstruction metrics
        for key in reversed(["valid", "solvable", "unique", "shortest_path", "resistance", "navigable_nodes"]):
            df.insert(0, f"R_{key}", reconstruction_metrics[key])
            if key in ["shortest_path", "resistance", "navigable_nodes"]:
                hist_data = torch.tensor(reconstruction_metrics[key]).to(pl_module.device, torch.float)
                hist_data = hist_data[~hist_data.isnan()].cpu().numpy()
                trainer.logger.experiment.log(
                    {
                        f"distributions/{tag}/R/{key}/{mode}": wandb.Histogram(hist_data),
                        "global_step"                   : trainer.global_step
                        })
        del reconstruction_metrics, hist_data
        logger.info(f"log_latent_embeddings(): logged graph metrics distributions to wandb.")

        if predictor_loss_unreg is not None:
            df.insert(0, "Predictor_loss_unreg", predictor_loss_unreg.tolist())
            df.insert(0, "y_hat", y_hat.tolist())
            df.insert(0, "y", y.tolist())
        if unweighted_elbos is not None:
            df.insert(0, "Unweighted_elbos", unweighted_elbos.tolist())

        if self.force_valid_reconstructions:
            df.insert(0, "RV:Reconstructions_Valid", [wandb.Image(x, mode="RGB") for x in reconstructed_imgs_force_valid])
        df.insert(0, "R:Reconstructions", [wandb.Image(x, mode="RGB") for x in reconstructed_imgs])
        if labels is not None:
            df.insert(0, "I:Inputs", [wandb.Image(x, mode="RGB") for x in input_imgs])
            df.insert(0, "Label_ids", labels.tolist())

        trainer.logger.experiment.log({
            f"tables/{tag}/{mode}": wandb.Table(
                dataframe=df
            ),
            "global_step": trainer.global_step
        })
        logger.info(f"log_latent_embeddings(): Logged dataframe tables/{tag}/{mode} with {len(reconstructed_imgs)} embeddings to wandb.")

    def log_latent_interpolation(self, trainer:pl.Trainer,
                                 pl_module:pl.LightningModule,
                                 tag: str,
                                 mode:str,
                                 num_samples:int=16,
                                 interpolations_per_samples:int=8,
                                 remove_dims:bool=True,
                                 dim_reduction_threshold:float=0.9,
                                 interpolation_scheme:str= 'linear',
                                 latent_sampling:bool=True):

        ### Slerp interpolate logic:
        # I. Compute distances between points in latent space and sort samples into pairs of lowest to highest distance.
        # II. For num_samples pairs:
        #   1. Spherical interpolation of the means : points = slerp(mean_1, mean_2)
        #   2. if latent sampling:
        #       Sample points around each interpolated mean according to a linear interpolation of the stds
        #       points = points + lerp(std1, std2) * randn()

        # New slerp interpolate logic:
        # I. Compute distances between points in latent space and sort samples into pairs of lowest to highest distance.
        # II. For num_samples pairs:
        #   1. normalise means to be unit vectors. mean_norm = mean / ||mean||
        #   2. Spherical interpolation on the unit sphere : points = slerp(mean_norm1, mean_norm2)
        #   3. Scale the points by the linear interpolation of the "radius" (mean): points = points * lerp(||mean1||, ||mean2||)
        #   4. if latent sampling:
        #       Sample points around each interpolated mean according to a linear interpolation of the stds
        #       points = points + lerp(std1, std2) * randn()

        logger.info(f"Progression: Entering log_latent_interpolation(), mode:{mode}")

        def interpolate_between_pairs(pairs:Tuple[int,int], data:torch.Tensor, num_interpolations:int,
                                      scheme:str)->torch.Tensor:

            interpolation_points = np.linspace(0, 1, num_interpolations + 2)[1:-1]
            interpolations = [[] for _ in pairs]
            for i, pair in enumerate(pairs):
                interpolations[i].append(data[pair[0]])
                for loc in interpolation_points:
                    if scheme == 'linear':
                        interp = lerp(loc, data[pair[0]], data[pair[1]])
                    elif scheme == 'polar':
                        interp = slerp(loc, data[pair[0]], data[pair[1]])
                    else:
                        raise RuntimeError(f"Interpolation scheme {interpolation_scheme} not recognised.")
                    interpolations[i].append(interp)
                interpolations[i].append(data[pair[1]])

            return torch.stack([torch.stack(row).to(data) for row in interpolations]).to(data)

        # Get data
        Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
        if mode == "val":
            mean = self.validation_step_outputs["mean"]
            std = self.validation_step_outputs["std"]
        elif mode == "predict":
            mean = self.predict_step_outputs["mean"]
            std = self.predict_step_outputs["std"]
        elif mode == "prior":
            mean = torch.randn(self.num_generated_samples, Z_dim).to(pl_module.device)
            std = None

        # Remove dimensions with average standard deviation of > dim_reduction_threshold (e.g. uninformative)
        if remove_dims and mode!="prior":
            kept_dims = torch.where(std < dim_reduction_threshold)[-1]
            kept_dims = torch.unique(kept_dims)
            trainer.logger.log_metrics({f"metric/latent_space/dim_info/{tag}/{mode}": len(kept_dims)},
                                       step=trainer.global_step)
            mean = mean[..., kept_dims]
            std = std[..., kept_dims]

        # Calculate and sort distances between latents
        if interpolation_scheme == 'linear':
            latents_dist = torch.cdist(mean, mean, p=2)
            eps = 5e-3
        elif interpolation_scheme == 'polar':
            latents_dist = cdist_polar(mean, mean)
            eps = 5e-3
        else:
            raise RuntimeError(f"Interpolation scheme {interpolation_scheme} not recognised.")

        latents_dist[latents_dist <= eps] = float('inf')
        dist, pair_indices = latents_dist.min(axis=0)
        dist = dist.cpu().numpy()
        pair_indices = pair_indices.cpu().numpy()
        distances = {}
        for ind_a, ind_b in enumerate(pair_indices):
            if ind_a != ind_b:
                key = (min(ind_a, ind_b), max(ind_a, ind_b))
                distances[key] = dist[ind_a]

        # Ensure we don't compute more interpolations than desired
        if len(distances.keys()) > num_samples:
            distances = {k : v for k, v in
                         zip(list(distances.keys())[:num_samples], list(distances.values())[:num_samples])}

        # sort the distances in ascending order
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

        if interpolation_scheme in ['linear','polar']:
            #1.
            interp_Z = interpolate_between_pairs(list(distances.keys()), mean, interpolations_per_samples, scheme=interpolation_scheme)
            #2.
            if latent_sampling and std is not None:
                interp_std = interpolate_between_pairs(list(distances.keys()), std, interpolations_per_samples, 'linear')
                interp_Z = interp_Z + interp_std * torch.randn_like(interp_Z)
                del interp_std
        # Not used at the moment
        elif interpolation_scheme == 'unit_sphere':
            #1.
            mean_norm = torch.linalg.norm(mean, dim=1)
            mean_normalised = mean / mean_norm.unsqueeze(-1)

            #2.
            interp_Z = interpolate_between_pairs(distances.keys(), mean_normalised, interpolations_per_samples, interpolation_scheme)

            #3.
            interp_mean_norms = interpolate_between_pairs(distances.keys(), mean_norm, interpolations_per_samples, 'linear')
            interp_Z = interp_Z * interp_mean_norms.unsqueeze(-1)

            #4.
            if latent_sampling and std is not None:
                interp_std = interpolate_between_pairs(distances.keys(), std, interpolations_per_samples, 'linear')
                interp_Z = interp_Z + interp_std * torch.randn_like(interp_Z)
                del interp_std

            del interp_mean_norms, mean_norm, mean_normalised
        else:
            raise RuntimeError(f"Interpolation scheme {interpolation_scheme} not recognised.")
        del mean, std

        # reassembly
        if remove_dims and mode!="prior":
            temp = interp_Z.clone().to(device=pl_module.device)
            interp_Z = torch.zeros(*interp_Z.shape[:-1], Z_dim,
                                                      dtype=torch.float).to(pl_module.device)
            interp_Z[..., kept_dims] = temp
            del temp

        to_log = {
            "distances": list(distances.values())
            }

        df = pd.DataFrame.from_dict(to_log)

        trainer.logger.experiment.log({
            f"tables/{tag}/distances/{mode}": wandb.Table(
                dataframe=df
            ),
            "global_step": trainer.global_step
        })

        interp_Z = interp_Z.reshape(-1, interp_Z.shape[-1])
        self.log_latent_embeddings(trainer, pl_module, "latent_space/interpolation", mode=mode, outputs={"Z":interp_Z},
                                   interpolation=True)

        # Obtain the interpolated samples
        logits = {}
        if self.dataset_cfg.encoding == "minimal":
            logits["logits_A"], logits["logits_Fx"] = pl_module.decoder(interp_Z)
        elif self.dataset_cfg.encoding == "dense":
            logits["logits_Fx"] = pl_module.decoder(interp_Z)
        else:
            raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not implemented.")
        del interp_Z

        imgs = self.obtain_imgs(logits, pl_module)
        num_rows = int(imgs.shape[0] // len(distances.keys()))

        self.log_images(trainer, f"interpolated_samples/{tag}/{mode}", imgs, mode="RGB", nrow=num_rows)

    def log_prior_sampling(self, trainer, pl_module, tag=None):
        logger.info(f"Progression: Entering log_prior_sampling(), tag:{tag}")
        tag = f"generated/prior/{tag}" if tag is not None else "generated/prior"
        Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
        Z = torch.randn(1, self.num_generated_samples, Z_dim).to(device=pl_module.device)
        logits = {}
        if self.dataset_cfg.encoding == "minimal":
            logits["logits_A"], logits["logits_Fx"] = [f.mean(dim=0) for f in pl_module.decoder(Z)]
            to_log = {
            f'metric/entropy/A/{tag}': pl_module.decoder.entropy_A(logits["logits_A"]),
            f'metric/entropy/Fx/{tag}': pl_module.decoder.entropy_Fx(logits["logits_Fx"]).sum()}
        elif self.dataset_cfg.encoding == "dense":
            logits["logits_Fx"] = pl_module.decoder(Z).mean(dim=0)
            to_log = {f'metric/entropy/Fx/{tag}': pl_module.decoder.entropy_Fx(logits["logits_Fx"]).sum()}
        else:
            raise ValueError(f"Encoding {self.dataset_cfg.encoding} not recognised.")
        del Z
        trainer.logger.log_metrics(to_log, trainer.global_step)

        for key in logits.keys():
            logits[key] = logits[key][:self.num_image_samples]
        generated_imgs = self.obtain_imgs(outputs=logits, pl_module=pl_module)
        self.log_images(trainer, tag, generated_imgs, mode="RGB")
        self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag=tag,
                               outputs=logits)

    # Utility methods

    def clear_stored_batches(self):
        self.batches_prepared = False
        self.validation_batch = deepcopy(self.batch_template)
        self.predict_batch = deepcopy(self.batch_template)
        self.validation_step_outputs = deepcopy(self.outputs_template)
        self.predict_step_outputs = deepcopy(self.outputs_template)
        logger.info(f"Cleared all stored batches.")

    def obtain_model_outputs(self, graphs, pl_module, num_samples, num_var_samples=1):
        logger.debug(f"obtain_model_outputs(): num_samples:{num_samples}, num_var_samples:{num_var_samples}")
        outputs = \
            pl_module.all_model_outputs_pathwise(graphs, num_samples=num_var_samples)
        for key in outputs.keys():
            outputs[key] = outputs[key][:num_samples]

        return outputs

    @staticmethod
    def prepare_stored_batch(batch_dict, output_dict, max_num_samples) -> bool:
        logger.info(f"Preparing the stored batches...")
        assert len(batch_dict["graphs"]) == len(output_dict["logits_Fx"]), "Error in prepare_stored_batches(). The number of stored batches and outputs must be the same"
        logger.info(f"Number of stored batches: {len(batch_dict['graphs'])}")
        logger.info(f"Number of stored batched outputs: {len(output_dict['loss'])}")
        def flatten(l):
            return [item for sublist in l for item in sublist]

        unbatched_graphs = [dgl.unbatch(g) for g in batch_dict["graphs"]]
        batch_dict["graphs"] = flatten(unbatched_graphs)
        batch_dict["label_ids"] = torch.cat(batch_dict["label_ids"])

        logger.info(f"Preparing the stored outputs...")
        for key in output_dict.keys():
            output_dict[key] = torch.cat(output_dict[key])

        for dict in batch_dict, output_dict:
            for key in dict.keys():
                dict[key] = dict[key][:max_num_samples]

        logger.info(f"Successfully unpacked the batches and outputs. Number of stored datapoints: {len(batch_dict['graphs'])}")
        return True

    @staticmethod
    def store_batch(batch_dict, output_dict, batch, outputs):

        batch_dict["graphs"].append(batch[0])
        batch_dict["label_ids"].append(batch[1])

        for key in output_dict.keys():
            try:
                output_dict[key].append(outputs[key])
            except KeyError:
                logger.warning(f"Key {key} not found in model outputs.")

        logger.debug(f"Successfully stored batch.")

    def obtain_imgs(self, outputs, pl_module:pl.LightningModule) -> torch.Tensor:
        """

        :param outputs:
        :param pl_module:
        :return: images: torch.Tensor of shape (B, C, H, W), C=3 [dense encoding] or C=4 [old encoding]
        """
        assert self.dataset_cfg.data_type == "graph", "Error in obtain_imgs(). This method is only valid for graph datasets."

        if self.dataset_cfg.encoding == "dense":
            mode_Fx = pl_module.decoder.param_m(outputs["logits_Fx"])
            if self.logging_cfg.rendering == "minigrid": #Not sure this should be the decider
                grids = tr.Nav2DTransforms.dense_features_to_minigrid(mode_Fx, node_attributes=pl_module.decoder.attributes)
                images = tr.Nav2DTransforms.minigrid_to_minigrid_render(grids, tile_size=self.logging_cfg.tile_size)
            else:
                raise NotImplementedError(f"Rendering method {self.logging_cfg.rendering} not implemented for dense graph encoding.")
        elif self.dataset_cfg.encoding == "minimal":
            assert self.logging_cfg.rendering != "minigrid", "Error in obtain_imgs(). Minigrid rendering is not implemented for minimal graph encoding."
            logger.warning(f"Warning {self.logging_cfg.rendering} is Deprecated. Correct behavior not guaranteed.")
            mode_probs = pl_module.decoder.param_m((outputs["logits_A"], outputs["logits_Fx"]))
            reconstructed_gws = self.encode_graph_to_gridworld(mode_probs, attributes=pl_module.decoder.attributes)
            images = self.gridworld_to_img(reconstructed_gws)
        else:
            raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not implemented.")

        return images

    def log_images(self, trainer:pl.Trainer, tag:str, images: torch.Tensor, captions:List[str]=None, mode:str=None, nrow:int=8):
        """
        Log images to Wandb.
        :param trainer:
        :param tag:
        :param images: [B, C, H, W]
        :param captions:
        :param mode:
        :param nrow:
        """


        logger.info(f"log_images(): Saving {len(images)} images as {1} grid with {nrow} rows - mode:{mode}, tag:{tag}")
        if mode == "RGBA":
            images = rgba2rgb(images)
        elif mode == "RGB":
            pass
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        img_grid = torchvision.utils.make_grid(self.resize_transform(images).to(torch.float), nrow=nrow, normalize=True)
        #TODO: reimplement captions
        if captions is None:
            captions = [None] * len(images)

        trainer.logger.experiment.log({
            f"images/{tag}": wandb.Image(img_grid), #[wandb.Image(img_grid, caption=c, mode=mode) for x, c in zip(images, captions)],
            "global_step": trainer.global_step
        })

    def log_prob_heatmaps(self, trainer, pl_module, tag, outputs):
        logger.info(f"Progression: Entering log_prob_heatmaps(), tag:{tag}")
        grid_dim = int(np.sqrt(self.dataset_cfg.max_nodes))  # sqrt(num_nodes)
        if self.dataset_cfg.encoding == "dense":
            logits_Fx = pl_module.decoder.param_p(outputs["logits_Fx"])
            heatmap_layout = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("active")], grid_dim)
        elif self.dataset_cfg.encoding == "minimal":
            logits_A, logits_Fx = pl_module.decoder.param_p((outputs["logits_A"], outputs["logits_Fx"]))
            heatmap_layout = self.prob_heatmap_A(logits_A, grid_dim)
        else:
            raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not implemented.")

        heatmap_start = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("start")], grid_dim)
        heatmap_goal = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("goal")], grid_dim)

        self.log_images(trainer, tag + "/prob_heatmap/start", heatmap_start, mode="RGBA")
        self.log_images(trainer, tag + "/prob_heatmap/goal", heatmap_goal, mode="RGBA")
        self.log_images(trainer, tag + "/prob_heatmap/layout", heatmap_layout, mode="RGBA")

    def prob_heatmap_fx(self, probs_fx, grid_dim):

        assert len(probs_fx.shape) == 2

        probs_fx = probs_fx / probs_fx.amax(dim=[i for i in range(1, len(probs_fx.shape))], keepdim=True)
        probs_fx = probs_fx.reshape(probs_fx.shape[0], grid_dim, grid_dim)
        heat_map = torch.zeros((*probs_fx.shape, 4)) # (B, H, W, C), C=RGBA
        heat_map[..., 0] = 1  # all values will be shades of red
        heat_map[..., 3] = probs_fx
        heat_map = einops.rearrange(heat_map, 'b h w c -> b c h w')

        heat_map = self.node_to_gw_mapping(heat_map) #add zeros in between for inactive nodes

        return heat_map

    def prob_heatmap_A(self, probs_A, grid_dim):

        assert len(probs_A.shape) == 2 #check A is flattened

        probs_A = probs_A.reshape(probs_A.shape[0], -1, 2)

        flip_bits = tr.FlipBinaryTransform()
        threshold = .5
        layout_dim = int(grid_dim * 2 + 1)
        layout_gw = tr.Nav2DTransforms.encode_reduced_adj_to_gridworld_layout(probs_A, (layout_dim, layout_dim), probalistic_mode=True, prob_threshold=threshold)

        heat_map = torch.zeros((*layout_gw.shape, 4)).to(layout_gw)  # (B, H, W, C), C=RGBA
        heat_map[..., 0][layout_gw >= threshold] = 1 #val >= threshold go to red channel
        heat_map[..., 3][layout_gw >= threshold] = (layout_gw[layout_gw >= threshold] - threshold)*2 #set transparency according to prob rescaled from [treshold,1] to [0,1]
        heat_map[..., 2][layout_gw < threshold] = 1 #val <= threshold go to red channel
        heat_map[..., 3][layout_gw < threshold] = flip_bits((layout_gw[layout_gw < threshold])*2)  #set transparency according to prob rescaled from [0,treshold] to [0,1], then flipped to [1,0]
        heat_map = einops.rearrange(heat_map, 'b h w c -> b c h w')

        return heat_map

    def encode_graph_to_gridworld(self, graphs, attributes):
        raise NotImplementedError("Deprecated")
        return tr.Nav2DTransforms.encode_graph_to_gridworld(graphs, attributes, self.used_attributes)

    def gridworld_to_img(self, gws):
        """
        :param gws: [B, C, H, W]
        :param gws:
        :return: imgs: rgb image tensor with dimensions [B, C, H, W]
        """
        colors = {
            'wall': [0., 0., 0., 1.],   #black
            'empty': [0., 0., 0., 0.],  #white
            'start': [0., 0., 1., 1.],  #blue
            'goal': [0., 1., 0., 1.],   #green
            # "bright_red" : [1, 0, 0, 1],
            # "light_red"  : [1, 0, 0, 0.1],
            # "bright_blue": [0, 0, 1, 1],
            # "light_blue" : [0, 0, 1, 0.1],
        }

        assert tr.OBJECT_TO_CHANNEL_AND_IDX["wall"][1] != 0 or tr.OBJECT_TO_CHANNEL_AND_IDX["empty"][1] != 0
        if tr.OBJECT_TO_CHANNEL_AND_IDX["empty"][1] != 0:
            colors.pop("wall")
        else:
            colors.pop("empty")

        imgs = torch.zeros((gws.shape[0], 4, *gws.shape[-2:])).to(gws.device)

        for feat in colors.keys():
            mask = gws[:, tr.OBJECT_TO_CHANNEL_AND_IDX[feat][0], ...] == tr.OBJECT_TO_CHANNEL_AND_IDX[feat][1]
            color = torch.tensor(colors[feat]).to(gws)
            cm = torch.einsum('c, b h w -> b c h w', color, mask)
            imgs[cm>0] = cm[cm>0]

        imgs = rgba2rgb(imgs)

        return imgs

    def normalise_metrics(self, metrics, device=torch.device("cpu")):

        for key in metrics.keys():
            if key in self.label_descriptors_config.keys():
                if isinstance(metrics[key], list):
                    metrics[key] = torch.tensor(metrics[key]).to(device, torch.float)
                metrics[key] /= self.label_descriptors_config[key].normalisation_factor
                if isinstance(metrics[key], torch.Tensor):
                    metrics[key] = metrics[key].tolist()

        return metrics