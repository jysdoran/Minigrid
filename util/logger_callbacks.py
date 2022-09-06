import logging

import dgl
import networkx as nx
import numpy as np
import torchvision
from typing import List, Union, Tuple, Optional, Any, Dict
import torch
import pytorch_lightning as pl
import wandb
import einops
import pandas as pd

from data_generators import OBJECT_TO_CHANNEL_AND_IDX
import util.transforms as tr
from util.graph_metrics import compute_metrics
from util.util import check_unique, cdist_polar, lerp, slerp, rgba2rgb
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
        self.used_attributes = logging_cfg.attribute_to_gw_encoding
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
        self.outputs_template = {
            "loss": [],
            "elbos": [],
            "unweighted_elbos": [],
            "neg_cross_entropy_A": [],
            "neg_cross_entropy_Fx": [],
            "kld": [],
            "predictor_loss": [],
            "predictor_loss_unreg": [],
            "y": [],
            "y_hat": [],
            "logits_A": [],
            "logits_Fx": [],
            "std": [],
            "mean": [],
            }

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
            self.gw[key] = self.encode_graph_to_gridworld(self.graphs[key], self.attributes).to(device)
            self.imgs[key] = self.gridworld_to_img(self.gw[key][:self.num_image_samples]).to(device)

        self.resize_transform = torchvision.transforms.Resize((100, 100),
                                               interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    # Main logging logic

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        try:
            self.log_images(trainer, "dataset/train", self.imgs["train"], self.labels["train"], mode="RGBA")
            self.log_images(trainer, "dataset/val", self.imgs["val"], self.labels["val"], mode="RGBA")
            self.log_images(trainer, "dataset/test", self.imgs["test"], self.labels["test"], mode="RGBA")
        except KeyError as e:
            logger.info(f"{e} dataset was not supplied in the samples provided to the GraphVAELogger")

    def on_validation_epoch_end(self, trainer, pl_module):
        _ = GraphVAELogger.prepare_stored_batch(self.predict_batch, self.predict_step_outputs, self.num_stored_samples)
        self.batches_prepared = GraphVAELogger.prepare_stored_batch(self.validation_batch, self.validation_step_outputs, self.num_stored_samples)
        self.log_epoch_metrics(trainer, pl_module, self.predict_step_outputs, "predict")
        self.log_epoch_metrics(trainer, pl_module, self.validation_step_outputs, "val")
        if "train" in self.graphs.keys():
            outputs = self.obtain_model_outputs(self.graphs["train"], pl_module, num_samples=self.num_image_samples, num_var_samples=1)
            reconstructed_imgs_train = self.obtain_imgs(outputs["logits_A"], outputs["logits_Fx"], pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["train"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/train", reconstructed_imgs_train, captions=captions, mode="RGBA")
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/train", logits_A=outputs["logits_A"], logits_Fx=outputs["logits_Fx"])

            #TODO just to see. remove later possibly
            outputs = self.obtain_model_outputs(self.graphs["train"], pl_module, num_samples=self.num_image_samples, num_var_samples=self.num_variational_samples_logging)
            self.log_epoch_metrics(trainer, pl_module, outputs, f"predict_{self.num_variational_samples_logging}_var_samples")
            reconstructed_imgs_train = self.obtain_imgs(outputs["logits_A"], outputs["logits_Fx"], pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["train"], outputs["unweighted_elbos"])]
            self.log_images(trainer, f"reconstructions/train/{self.num_variational_samples_logging}_var_sample", reconstructed_imgs_train, captions=captions, mode="RGBA")
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag=f"reconstructions/train/{self.num_variational_samples_logging}_var_sample", logits_A=outputs["logits_A"], logits_Fx=outputs["logits_Fx"])

        if "val" in self.graphs.keys():
            outputs = self.obtain_model_outputs(self.graphs["val"], pl_module, num_samples=self.num_image_samples, num_var_samples=1)

            reconstructed_imgs_val = self.obtain_imgs(outputs["logits_A"], outputs["logits_Fx"], pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["val"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/val", reconstructed_imgs_val, captions=captions, mode="RGBA")
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/val", logits_A=outputs["logits_A"], logits_Fx=outputs["logits_Fx"])

        self.log_prior_sampling(trainer, pl_module, tag=None)

    def log_epoch_metrics(self, trainer, pl_module, outputs, mode):

        if not isinstance(outputs, dict):
            # assert len(outputs) == 6, "GraphVAElogger.log_epoch_metrics() - incorrect number of outputs provided. " \
            #                           f"Expected {6} outputs (elbos, unweighted_elbos, logits_A, logits_Fx, mean, std)," \
            #                           f"got {len(outputs)} instead."
            # elbos, unweighted_elbos, logits_A, logits_Fx, mean, std, y_hat, predictor_loss_unreg = outputs
            # loss = -elbos.mean()
            raise NotImplementedError("GraphVAElogger.log_epoch_metrics() - only handles outputs in dict format")

        to_log = {
            f"metric/mean/std/{mode}"   : torch.linalg.norm(outputs["mean"], dim=-1).std().item(),
            f"metric/sigma/mean/{mode}" : torch.square(outputs["std"]).sum(axis=-1).sqrt().mean().item() / outputs["std"].shape[-1],
            f'metric/entropy/A/{mode}'  : pl_module.decoder.entropy_A(outputs["logits_A"]).sum(),
            f'metric/entropy/Fx/{mode}' : pl_module.decoder.entropy_A(outputs["logits_Fx"]).sum(),
            }
        for key in outputs.keys():
            to_log[f"metric/{key}/{mode}"] = outputs[key].mean()

        trainer.logger.log_metrics(to_log, step=trainer.global_step)

        flattened_logits_A = torch.flatten(pl_module.decoder.param_pA(outputs["logits_A"]))
        flattened_logits_Fx = torch.flatten(pl_module.decoder.param_pFx(outputs["logits_Fx"]))
        trainer.logger.experiment.log(
            {f"distributions/logits/A/{mode}": wandb.Histogram(flattened_logits_A.to("cpu")),
             "global_step": trainer.global_step})
        trainer.logger.experiment.log(
            {f"distributions/logits/Fx/{mode}": wandb.Histogram(flattened_logits_Fx.to("cpu")),
             "global_step": trainer.global_step})


    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Progression: Entering on_train_end()")
        self.log_latent_embeddings(trainer, pl_module, "latent_space", mode="val")
        #TODO remove later
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
        #TODO remove later
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
            outputs = self.obtain_model_outputs(self.graphs["test"],
                                                                                            pl_module, num_samples=self.num_image_samples, num_var_samples=1)
            reconstructed_imgs = self.obtain_imgs(outputs["logits_A"], outputs["logits_Fx"], pl_module)
            captions = [f"Label:{l}, unweighted_elbo:{e}" for (l,e) in zip(self.labels["test"], outputs["unweighted_elbos"])]
            self.log_images(trainer, "reconstructions/test", reconstructed_imgs, captions=captions, mode="RGBA")

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

    def log_latent_embeddings(self, trainer, pl_module, tag, mode="val", outputs=None):

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
                    Z_dim = pl_module.hparams.configuration.shared_parameters.latent_dim
                    Z = torch.randn(self.num_generated_samples, Z_dim).to(device=pl_module.device)
            logits_A, logits_Fx = pl_module.decoder(Z)
            unweighted_elbos = None
            labels = None
            y_hat = pl_module.predictor(Z)
            reconstructed_graphs, start_nodes, goal_nodes, is_valid = \
                tr.Nav2DTransforms.encode_decoder_output_to_graph(logits_A, logits_Fx, pl_module.decoder,
                                                                  correct_A=True)
            y = pl_module.predictor.target_metric_fn(reconstructed_graphs, start_nodes, goal_nodes).to(Z.device)
            y = einops.repeat(y, 'b -> b 1') # (B,) -> (B,1)
            predictor_loss_fn = pl_module.predictor.loss_fn(reduction="none")
            predictor_loss_unreg = predictor_loss_fn(y_hat, y)
        elif mode == "val":
            Z = self.validation_step_outputs["mean"][:self.num_embedding_samples]
            logits_A, logits_Fx = self.validation_step_outputs["logits_A"][:self.num_embedding_samples], self.validation_step_outputs["logits_Fx"][:self.num_embedding_samples]
            graphs = self.validation_batch["graphs"][:self.num_embedding_samples]
            labels = self.validation_batch["label_ids"][:self.num_embedding_samples]
            unweighted_elbos = self.validation_step_outputs["unweighted_elbos"][:self.num_embedding_samples]
            predictor_loss_unreg = self.validation_step_outputs["predictor_loss_unreg"][:self.num_embedding_samples]
            y_hat = self.validation_step_outputs["y_hat"][:self.num_embedding_samples]
            y = self.validation_step_outputs["y"][ :self.num_embedding_samples ]
        elif mode == "predict":
            Z = self.predict_step_outputs["mean"][:self.num_embedding_samples]
            logits_A, logits_Fx = self.predict_step_outputs["logits_A"][:self.num_embedding_samples], self.predict_step_outputs["logits_Fx"][:self.num_embedding_samples]
            graphs = self.predict_batch["graphs"][:self.num_embedding_samples]
            labels = self.predict_batch["label_ids"][:self.num_embedding_samples]
            unweighted_elbos = self.predict_step_outputs["unweighted_elbos"][:self.num_embedding_samples]
            predictor_loss_unreg = self.predict_step_outputs["predictor_loss_unreg"][:self.num_embedding_samples]
            y_hat = self.predict_step_outputs["y_hat"][:self.num_embedding_samples]
            y = self.predict_step_outputs["y"][:self.num_embedding_samples ]
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

        reconstructed_imgs = self.obtain_imgs(logits_A, logits_Fx, pl_module)

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
            reconstructed_imgs_force_valid = self.obtain_imgs(logits_A, mode_Fx, pl_module)
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
            df.insert(0, "RV:Reconstructions_Valid", [wandb.Image(x, mode="RGBA") for x in reconstructed_imgs_force_valid])
        df.insert(0, "R:Reconstructions", [wandb.Image(x, mode="RGBA") for x in reconstructed_imgs])
        if labels is not None:
            df.insert(0, "I:Inputs", [wandb.Image(x, mode="RGBA") for x in input_imgs])
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
        mean, std, Z = [None]*3
        Z_dim = pl_module.hparams.configuration.shared_parameters.latent_dim
        if mode == "val":
            mean = self.validation_step_outputs["mean"][:num_samples]
            std = self.validation_step_outputs["std"][:num_samples]
            if latent_sampling:
                rand = torch.randn(len(mean), Z_dim).to(mean)
                Z = mean + std * rand
            else:
                Z = mean
        elif mode == "predict":
            mean = self.predict_step_outputs["mean"][:num_samples]
            std = self.predict_step_outputs["std"][:num_samples]
            if latent_sampling:
                rand = torch.randn(len(mean), Z_dim).to(pl_module.device)
                Z = mean + std * rand
            else:
                Z = mean
        elif mode == "prior":
            Z = torch.randn(num_samples, Z_dim).to(pl_module.device)

        # Remove dimensions with average standard deviation of 1 (e.g. uninformative)
        if remove_dims and mode!="prior":
            kept_dims = torch.where(std < dim_reduction_threshold)[-1]
            kept_dims = torch.unique(kept_dims)
            trainer.logger.log_metrics({f"metric/latent_space/dim_info/{tag}/{mode}": len(kept_dims)},
                                       step=trainer.global_step)
            Z = Z[..., kept_dims]
            std = std[..., kept_dims]
            mean = mean[..., kept_dims]

        # Calculate and sort distances between latents
        if interpolation_scheme == 'linear':
            latents_dist = torch.cdist(Z, Z, p=2)
            eps = 5e-3
        if interpolation_scheme == 'polar':
            latents_dist = cdist_polar(Z, Z)
            eps = 5e-3

        latents_dist[latents_dist <= eps] = float('inf')
        dist, pair_indices = latents_dist.min(axis=0)
        dist = dist.cpu().numpy()
        pair_indices = pair_indices.cpu().numpy()
        distances = {}
        for ind_a, ind_b in enumerate(pair_indices):
            if ind_a != ind_b:
                key = (min(ind_a, ind_b), max(ind_a, ind_b))
                distances[key] = dist[ind_a]

        # sort the distances in ascending order
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

        interp_Z = interpolate_between_pairs(distances.keys(), Z, interpolations_per_samples, interpolation_scheme)
        if mode != "prior":
            interp_mean = interpolate_between_pairs(distances.keys(), mean, interpolations_per_samples, 'linear')
            interp_std = interpolate_between_pairs(distances.keys(), std, interpolations_per_samples, 'linear')
            interp_Z = (interp_Z * interp_std) + interp_mean
            del interp_mean, interp_std, std, mean
        del Z

        # reassembly
        if remove_dims and mode!="prior":
            temp = interp_Z.clone().to(device=pl_module.device)
            interp_Z = torch.zeros(*interp_Z.shape[:-1], Z_dim,
                                                      dtype=torch.float).to(pl_module.device)
            interp_Z[..., kept_dims] = temp
            del temp

        interp_Z = interp_Z.reshape(-1, interp_Z.shape[-1])
        self.log_latent_embeddings(trainer, pl_module, "latent_space/interpolation", mode="val", outputs={"Z":interp_Z})

        # Obtain the interpolated samples
        logits_A, logits_Fx = pl_module.decoder(interp_Z)
        del interp_Z

        imgs = self.obtain_imgs(logits_A, logits_Fx, pl_module)

        self.log_images(trainer, f"interpolated_samples/{tag}/{mode}", imgs, mode="RGBA", nrow=len(distances.keys()))

    def log_prior_sampling(self, trainer, pl_module, tag=None):
        logger.info(f"Progression: Entering log_prior_sampling(), tag:{tag}")
        tag = f"generated/prior/{tag}" if tag is not None else "generated/prior"
        Z_dim = pl_module.hparams.configuration.shared_parameters.latent_dim
        Z = torch.randn(1, self.num_generated_samples, Z_dim).to(device=pl_module.device)
        logits_A, logits_Fx = [f.mean(dim=0) for f in pl_module.decoder(Z)]
        del Z
        trainer.logger.log_metrics({
            f'metric/entropy/A/{tag}': pl_module.decoder.entropy_A(logits_A),
            f'metric/entropy/Fx/{tag}': pl_module.decoder.entropy_Fx(logits_Fx).sum()},
            trainer.global_step)

        generated_imgs = self.obtain_imgs(logits_A[:self.num_image_samples], logits_Fx[:self.num_image_samples], pl_module)
        self.log_images(trainer, tag, generated_imgs, mode="RGBA")
        self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag=tag, logits_A=logits_A[:self.num_image_samples], logits_Fx=logits_Fx[:self.num_image_samples])

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
        assert len(batch_dict["graphs"]) == len(output_dict["logits_A"]), "Error in prepare_stored_batches(). The number of stored batches and outputs must be the same"
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

    def obtain_imgs(self, logits_A:torch.Tensor, logits_Fx:torch.Tensor, pl_module:pl.LightningModule) -> torch.Tensor:

        mode_probs = pl_module.decoder.param_m((logits_A, logits_Fx))
        reconstructed_gws = self.encode_graph_to_gridworld(mode_probs, attributes=self.used_attributes)
        reconstructed_imgs = self.gridworld_to_img(reconstructed_gws)

        return reconstructed_imgs

    def log_images(self, trainer:pl.Trainer, tag:str, images: torch.Tensor, captions:List[str]=None, mode:str=None, nrow:int=8):
        logger.info(f"log_images(): Saving {len(images)} images as {1} grid with {nrow} rows - mode:{mode}, tag:{tag}")
        if mode == "RGBA":
            images = rgba2rgb(images)

            # TODO: add conversion to imgs, also should sitch the stored logits_A, logits_Fx (or not include them above)
            # also only get up to num_samples array size for Z.

        img_grid = torchvision.utils.make_grid(self.resize_transform(images), nrow=nrow)
        #TODO: reimplement captions
        if captions is None:
            captions = [None] * len(images)

        trainer.logger.experiment.log({
            f"images/{tag}": wandb.Image(img_grid), #[wandb.Image(img_grid, caption=c, mode=mode) for x, c in zip(images, captions)],
            "global_step": trainer.global_step
        })

    def log_prob_heatmaps(self, trainer, pl_module, tag, logits_A, logits_Fx):
        logger.info(f"Progression: Entering log_prob_heatmaps(), tag:{tag}")
        logits_A, logits_Fx = pl_module.decoder.param_p((logits_A, logits_Fx))
        grid_dim = int(np.sqrt(logits_Fx.shape[-2])) #sqrt(num_nodes)
        heatmap_start = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("start")], grid_dim)
        heatmap_goal = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("goal")], grid_dim)
        heatmap_layout = self.prob_heatmap_A(logits_A, grid_dim)
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

        # TODO: to revise for non reduced formulations
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
        return tr.Nav2DTransforms.encode_graph_to_gridworld(graphs, attributes, self.used_attributes)

    def gridworld_to_img(self, gws):
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

        assert OBJECT_TO_CHANNEL_AND_IDX["wall"][1] != 0 or OBJECT_TO_CHANNEL_AND_IDX["empty"][1] != 0
        if OBJECT_TO_CHANNEL_AND_IDX["empty"][1] != 0:
            colors.pop("wall")
        else:
            colors.pop("empty")

        imgs = torch.zeros((gws.shape[0], 4, *gws.shape[-2:])).to(gws.device)

        for feat in colors.keys():
            mask = gws[:, OBJECT_TO_CHANNEL_AND_IDX[feat][0], ...] == OBJECT_TO_CHANNEL_AND_IDX[feat][1]
            color = torch.tensor(colors[feat]).to(gws)
            cm = torch.einsum('c, b h w -> b c h w', color, mask)
            imgs[cm>0] = cm[cm>0]

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