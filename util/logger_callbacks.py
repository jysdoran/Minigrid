import copy
import logging

import dgl
import networkx as nx
import numpy as np
import torchvision
from typing import List, Union, Tuple, Optional, Any, Dict
from collections import defaultdict, OrderedDict
import torch
import pytorch_lightning as pl
import wandb
import einops
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from envs.multigrid.multigrid import Grid

from . import transforms as tr
from .graph_metrics import compute_metrics, get_non_nav_spl
from .util import check_unique, check_novel, get_node_features, cdist_polar, interpolate_between_pairs, rgba2rgb, dgl_to_nx
from copy import deepcopy
from maze_representations.data_loaders import GridNavDataModule

logger = logging.getLogger(__name__)


class GraphVAELogger(pl.Callback):
    def __init__(self,
                 data_module: GridNavDataModule,
                 logging_cfg,
                 dataset_cfg,
                 accelerator: str = "cpu"):

        super().__init__()
        device = torch.device("cuda" if accelerator == "gpu" else "cpu")
        self.logging_cfg = logging_cfg
        self.dataset_cfg = dataset_cfg

        self.data_module = data_module
        self.label_contents = data_module.target_contents
        self.label_descriptors_config = data_module.dataset_metadata.config.label_descriptors_config
        self.dataset_metadata = data_module.dataset_metadata
        self.force_valid_reconstructions = logging_cfg.force_valid_reconstructions
        self.num_stored_samples = logging_cfg.num_stored_samples
        self.num_image_samples = logging_cfg.num_image_samples
        self.num_embedding_samples = logging_cfg.num_embedding_samples
        self.num_generated_samples = logging_cfg.num_generated_samples
        self.num_variational_samples_logging = logging_cfg.num_variational_samples
        self.max_cached_batches = self.num_stored_samples // self.dataset_cfg.batch_size
        self.max_cached_batches = max(self.max_cached_batches, 1) #store at least one batch
        self.attributes = dataset_cfg.node_attributes

        graphs = []
        for (graph, label) in data_module.train:
            graphs.append(graph)
        self.train_dataset_features, _ = get_node_features(graphs, device=device)

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
        self.predict_aggregate_metrics = {"prob_moss": pd.DataFrame(columns=["source", "spl", "prob", "use"]),
                                          "prob_lava": pd.DataFrame(columns=["source", "spl", "prob", "use"]),
                                          "start_spl": pd.DataFrame(columns=["source", "spl", "count", "use"]),
                                          "spl_med":   pd.DataFrame(columns=["source", "spl", "count", "use"]),
                                          }

        try:
            for key in ["train", "val", "test"]:
                self.graphs[key], self.labels[key] = data_module.samples[key]
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
            if self.dataset_cfg.encoding == "dense":
                self.gw[key] = None
                if "images" in self.label_contents.keys():
                    self.imgs[key] = self.label_contents["images"][self.labels[key]][:self.num_image_samples]
                else:
                    self.imgs[key] = tr.Nav2DTransforms.dense_graph_to_minigrid_render(self.graphs[key], tile_size=16)[:self.num_image_samples]
            else:
                raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not supported.")
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
            }

        for key in outputs.keys():
            if key in ["logits_heads", "logits_Fx"]:
                continue
            to_log[f"metric/{key}/{mode}"] = outputs[key].mean()

        entropy = pl_module.decoder.entropy(outputs["logits_heads"])
        probs = pl_module.decoder.param_p(outputs["logits_heads"])
        for head in entropy:
            to_log[f'metric/entropy/heads/{head}/{mode}'] = entropy[head].mean()
            flattened_logits = torch.flatten(probs[head])
            trainer.logger.experiment.log(
                {
                    f"distributions/logits/Fx/{mode}": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step"                    : trainer.global_step
                    })

        trainer.logger.log_metrics(to_log, step=trainer.global_step)

        if "logits_A" in outputs.keys():
            to_log[f'metric/entropy/A/{mode}'] = pl_module.decoder.entropy_A(outputs["logits_A"]).mean()
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

        for metric, data in self.predict_aggregate_metrics.items():
            fig, ax = plt.subplots(figsize=(16, 14))
            if metric in ["prob_moss", "prob_lava"]:
                sns.histplot(data=data, x="spl", hue="source", weights="prob", discrete=True, ax=ax) #kde fit is bad
                ax.set_ylabel(metric, fontsize=20)
            elif metric in ["start_spl", "spl_med"]: #["start_loc"]:
                sns.histplot(data=data, x="spl", hue="source", discrete=True, ax=ax, kde=True, common_norm=False)
            else:
                raise NotImplementedError(f"Metric {metric} not implemented for plotting")
            ax.set_xlabel('shortest_path_length', fontsize=20)
            # plt.show() #only for debug, comment out
            fig = wandb.Image(fig) #plotly does not support seaborn
            wandb.log({f"chart_{metric}_predict":fig})

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
            outputs["logits_heads"] = pl_module.decoder(outputs["Z"])
        else:
            if mode == "prior":
                input_batch = {}
                assert outputs is None, "mode=prior with interpolation=false does not support passing outputs in."
                outputs = {}
                Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
                outputs["Z"] = torch.randn(self.num_generated_samples, Z_dim).to(device=pl_module.device)
                outputs["logits_heads"] = pl_module.decoder(outputs["Z"])
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
                    if isinstance(outputs[key], torch.Tensor):
                        outputs[key] = outputs[key][:self.num_embedding_samples]
                    elif isinstance(outputs[key], dict):
                        for subkey in outputs[key].keys():
                            outputs[key][subkey] = outputs[key][subkey][:self.num_embedding_samples]
                for key in input_batch.keys():
                    input_batch[key] = input_batch[key][:self.num_embedding_samples]
                outputs["Z"] = outputs["mean"]

        assert outputs["Z"].shape[0] == \
               outputs['logits_heads'][list(outputs['logits_heads'].keys())[0]].shape[0], \
            f"log_latent_embeddings(): Shape mismatch Z={outputs['Z'].shape} != " \
            f"logits_heads={outputs['logits_heads'][list(outputs['logits_heads'].keys())[0]].shape}"

        edge_config_nav = {'navigable': self.dataset_metadata.config.graph_edge_descriptors['navigable']}

        reconstructed_graphs = pl_module.decoder.to_graph(outputs["logits_heads"],
                                                          make_batch=False, masked=True, edge_config=edge_config_nav)
        reconstructed_graphs = tr.Nav2DTransforms.graph_to_grid_graph(reconstructed_graphs, level_info=self.dataset_metadata.level_info)
        reconstructed_imgs = self.obtain_imgs(outputs, pl_module)
        reconstruction_metrics = compute_metrics(reconstructed_graphs, labels=input_batch.get("label_ids"))
        rec_Fx = pl_module.decoder.to_graph_features(outputs["logits_heads"])
        reconstruction_metrics["unique"] = check_unique(rec_Fx)
        features_idx = [self.attributes.index(f) for f in rec_Fx.keys()]
        train_features = self.train_dataset_features[..., features_idx]
        rec_Fx = torch.stack([rec_Fx[k] for k in rec_Fx.keys()], dim=-1)
        novel_dataset = check_novel(rec_Fx, train_features).cpu()
        reconstruction_metrics["novel"] = novel_dataset & reconstruction_metrics["solvable"]
        frac_novel_not_repeated = torch.unique(rec_Fx[reconstruction_metrics["novel"]], dim=0).shape[0] / rec_Fx.shape[0]
        # solvable_graphs only
        solvable_graphs = [g for g, s in zip(reconstructed_graphs, reconstruction_metrics["solvable"]) if s]
        model_p_cn_cm = self.prob_moss_over_spl(solvable_graphs)
        model_p_cnn_cl = self.prob_lava_over_spl(solvable_graphs)
        model_spl_start, model_spl_med = self.spl_start(solvable_graphs, normalise=None)

        count_threshold = 100  # TODO: make param
        if input_batch.get("graphs") is not None:
            input_imgs = tr.Nav2DTransforms.dense_graph_to_minigrid_render(input_batch["graphs"],
                                                                     tile_size=self.logging_cfg.tile_size)
            dataset_nav_graphs = [self.label_contents['edge_graphs']['navigable'][t.item()] for t in input_batch["label_ids"]]
            # refers to graphs that have solvable reconstructions.
            dataset_solvable_graphs = [g for g, s in zip(dataset_nav_graphs, reconstruction_metrics["solvable"]) if s]
            dataset_solvable_graphs = tr.Nav2DTransforms.graph_to_grid_graph(dataset_solvable_graphs, level_info=self.dataset_metadata.level_info)

            solvable_targets = [t for t, s in zip(input_batch.get("label_ids"), reconstruction_metrics["solvable"]) if s]
            dataset_p_cn_cm = self.prob_moss_over_spl(dataset_solvable_graphs, targets=solvable_targets)
            dataset_p_cnn_cl = self.prob_lava_over_spl(dataset_solvable_graphs, targets=solvable_targets)
            dataset_spl_start, dataset_spl_med = self.spl_start(dataset_solvable_graphs, targets=solvable_targets)
            for spl, count in dataset_p_cn_cm[1].items():
                if count < count_threshold:
                    spl_nav_threshold = spl #NOTE: spl < spl_threshold and not spl <= spl_threshold
                    break
            for spl, count in dataset_p_cnn_cl[1].items():
                if count < count_threshold:
                    spl_non_nav_threshold = spl
                    break
            del dataset_nav_graphs, dataset_solvable_graphs, solvable_targets
        else:
            for spl, count in model_p_cn_cm[1].items():
                if count < count_threshold:
                    spl_nav_threshold = spl #NOTE: spl < spl_threshold and not spl <= spl_threshold
                    break
            for spl, count in model_p_cnn_cl[1].items():
                if count < count_threshold:
                    spl_non_nav_threshold = spl
                    break

        if mode == "predict":
            if not interpolation:
                aggregate_data = {"dataset": {"prob_moss": dataset_p_cn_cm,
                                    "prob_lava": dataset_p_cnn_cl,
                                    "start_spl": dataset_spl_start,
                                    "spl_med": dataset_spl_med}}
                id_mode = "rec"
            else:
                aggregate_data = {}
                id_mode = "interp"

            aggregate_data["model"] = {"prob_moss": model_p_cn_cm,
                             "prob_lava": model_p_cnn_cl,
                             "start_spl": model_spl_start,
                             "spl_med" : model_spl_med,
                             #"start_loc" : {"spl_start": model_spl_start, "spl_med": model_spl_med}
                                       }

            for data_source, data_dict in aggregate_data.items(): #i.e. model, {"prob_X": data_X, "prob_Y": data_Y}
                frames = defaultdict(list)
                if data_source == "dataset":
                    source = f"dataset"
                elif data_source == "model":
                    source = f"{id_mode}_model"
                else:
                    raise ValueError(f"Invalid data source {data_source}")
                for query, data in data_dict.items(): #i.e "prob_X": data_X
                    if query in ["prob_moss", "prob_lava"]:
                        use_data = [1 if spl < spl_nav_threshold else 0 for spl in data[0].keys()]
                        # TODO: hacky, a better way would be to log all points in a df and then compute the histogram by filtering the spl.
                        probs = (np.array(list(data[0].values())) * np.array(use_data)).tolist()
                        if query == "prob_moss":
                            kw1, kw2 = "nav", "moss"
                        elif query == "prob_lava":
                            kw1, kw2 = "non_nav", "lava"
                        frames[query].append({"source": [source]*len(data[0]),
                                     "spl": list(data[0].keys()),
                                     "prob": probs,
                                     f"count_{kw1}": list(data[1].values()),
                                     f"count_{kw2}": list(data[2].values()),
                                     "use": use_data,
                                    })
                    elif query in ["start_spl", "spl_med"]:
                        frames[query].append({"source": [source]*len(data),
                                     "spl": data.tolist(),
                                     "count": [1]*len(data),
                                     "use": [1]*len(data),
                                    })
                    else:
                        raise ValueError(f"Invalid query {query}")
                for query, frame in frames.items():
                    frame = [pd.DataFrame(f) for f in frame]
                    self.predict_aggregate_metrics[query] = pd.concat([self.predict_aggregate_metrics[query], *frame])


        del outputs["logits_heads"], reconstructed_graphs, rec_Fx, train_features, solvable_graphs

        if self.label_descriptors_config is not None:
            self.normalise_metrics(reconstruction_metrics, device=pl_module.device)
            logger.info(f"log_latent_embeddings(): graph metrics normalised.")

        # Log average metrics
        to_log = {}
        for key in reconstruction_metrics.keys():
            data = reconstruction_metrics[key].to(pl_module.device, torch.float)
            data = data[~torch.isnan(data)].mean()
            to_log[f'metric/{tag}/task_metric/R/{key}/{mode}'] = data
        to_log[f'metric/{tag}/task_metric/R/novel_unique/{mode}'] = frac_novel_not_repeated
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
                data = [self.label_contents[input_property][i] for i in input_batch["label_ids"].tolist()]
                df.insert(0, f"I_{input_property}", data)


        # Store Reconstruction metrics in the embeddings table
        for key in reversed(["valid", "solvable", "unique", "novel", "shortest_path", "resistance", "navigable_nodes"]):
            df.insert(0, f"R_{key}", reconstruction_metrics[key].tolist())
            if key in ["shortest_path", "resistance", "navigable_nodes"]:
                hist_data = reconstruction_metrics[key].to(pl_module.device, torch.float)
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

        # New slerp interpolate logic (unit-sphere, not used in _dcd):
        # I. Compute distances between points in latent space and sort samples into pairs of lowest to highest distance.
        # II. For num_samples pairs:
        #   1. normalise means to be unit vectors. mean_norm = mean / ||mean||
        #   2. Spherical interpolation on the unit sphere : points = slerp(mean_norm1, mean_norm2)
        #   3. Scale the points by the linear interpolation of the "radius" (mean): points = points * lerp(||mean1||, ||mean2||)
        #   4. if latent sampling:
        #       Sample points around each interpolated mean according to a linear interpolation of the stds
        #       points = points + lerp(std1, std2) * randn()

        logger.info(f"Progression: Entering log_latent_interpolation(), mode:{mode}")

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
            latents_dist = torch.cdist(mean.cpu(), mean.cpu(), p=2)
            eps = 5e-3
        elif interpolation_scheme == 'polar':
            latents_dist = cdist_polar(mean.cpu(), mean.cpu())
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
        pairs = list(distances.keys())

        if interpolation_scheme in ['linear','polar']:
            #1.
            interp_Z = interpolate_between_pairs(pairs, mean, interpolations_per_samples, scheme=interpolation_scheme)
            #2.
            if latent_sampling and std is not None:
                interp_std = interpolate_between_pairs(pairs, std, interpolations_per_samples, 'linear')
                interp_Z = interp_Z + interp_std * torch.randn_like(interp_Z)
                del interp_std
        # Not used at the moment
        elif interpolation_scheme == 'unit_sphere':
            #1.
            mean_norm = torch.linalg.norm(mean, dim=1)
            mean_normalised = mean / mean_norm.unsqueeze(-1)

            #2.
            interp_Z = interpolate_between_pairs(pairs, mean_normalised, interpolations_per_samples, interpolation_scheme)

            #3.
            interp_mean_norms = interpolate_between_pairs(pairs, mean_norm, interpolations_per_samples, 'linear')
            interp_Z = interp_Z * interp_mean_norms.unsqueeze(-1)

            #4.
            if latent_sampling and std is not None:
                interp_std = interpolate_between_pairs(pairs, std, interpolations_per_samples, 'linear')
                interp_Z = interp_Z + interp_std * torch.randn_like(interp_Z)
                del interp_std

            del interp_mean_norms, mean_norm, mean_normalised
        else:
            raise RuntimeError(f"Interpolation scheme {interpolation_scheme} not recognised.")

        # reassembly
        if remove_dims and mode!="prior":
            temp = interp_Z.clone().to(device=pl_module.device)
            interp_Z = torch.zeros(*interp_Z.shape[:-1], Z_dim, dtype=torch.float).to(pl_module.device)
            interp_Z[..., kept_dims] = temp
            temp = mean.clone().to(device=pl_module.device)
            mean = torch.zeros(*mean.shape[:-1], Z_dim, dtype=torch.float).to(pl_module.device)
            mean[..., kept_dims] = temp
            temp = std.clone().to(device=pl_module.device)
            std = torch.ones(*std.shape[:-1], Z_dim, dtype=torch.float).to(pl_module.device)
            std[..., kept_dims] = temp
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

        self.log_latent_embeddings(trainer, pl_module, "latent_space/interpolation", mode=mode, outputs={"Z":interp_Z},
                                   interpolation=True)

        # Obtain the interpolated samples + original pairs
        all_Z = []
        interp_Z = interp_Z.reshape(len(pairs), -1, Z_dim)
        for i, pair in enumerate(pairs):
            all_Z.append(mean[pair[0]])
            for latent in interp_Z[i]:
                all_Z.append(latent)
            all_Z.append(mean[pair[1]])
        all_Z = torch.stack(all_Z)

        del mean, std, interp_Z

        logits = {}
        if self.dataset_cfg.encoding == "dense":
            logits["logits_heads"] = pl_module.decoder(all_Z)
        else:
            raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not implemented.")
        del all_Z

        imgs = self.obtain_imgs(logits, pl_module)
        num_rows = int(imgs.shape[0] // len(pairs))

        self.log_images(trainer, f"interpolated_samples/{tag}/{mode}", imgs, mode="RGB", nrow=num_rows)

    def log_prior_sampling(self, trainer, pl_module, tag=None):
        logger.info(f"Progression: Entering log_prior_sampling(), tag:{tag}")
        tag = f"generated/prior/{tag}" if tag is not None else "generated/prior"
        Z_dim = pl_module.hparams.config_model.shared_parameters.latent_dim
        Z = torch.randn(1, self.num_generated_samples, Z_dim).to(device=pl_module.device)
        if self.dataset_cfg.encoding == "dense":
            logits = pl_module.decoder(Z)
            for key, val in logits.items():
                logits[key] = val.mean(dim=0)
            entropy = pl_module.decoder.entropy(logits)
            to_log = {}
            for head in entropy:
                to_log[f'metric/entropy/heads/{head}/{tag}'] = entropy[head].sum()
        else:
            raise ValueError(f"Encoding {self.dataset_cfg.encoding} not recognised.")
        del Z
        trainer.logger.log_metrics(to_log, trainer.global_step)

        for key in logits.keys():
            logits[key] = logits[key][:self.num_image_samples]
        generated_imgs = self.obtain_imgs(outputs={"logits_heads": logits}, pl_module=pl_module)
        self.log_images(trainer, tag, generated_imgs, mode="RGB")
        self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag=tag,
                               outputs={"logits_heads": logits})

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
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key][:num_samples]
            elif isinstance(outputs[key], dict):
                for subkey in outputs[key].keys():
                    outputs[key][subkey] = outputs[key][subkey][:num_samples]

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
            if isinstance(output_dict[key][0], torch.Tensor):
                output_dict[key] = torch.cat(output_dict[key])
            elif isinstance(output_dict[key][0], dict):
                stacked_values = {}
                for subkey in output_dict[key][0].keys():
                    stacked_values[subkey] = torch.cat([d[subkey] for d in output_dict[key]])
                output_dict[key] = stacked_values
            else:
                raise ValueError(f"Type {type(output_dict[key][0])} not recognised.")


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

    def obtain_imgs(self, outputs, pl_module:pl.LightningModule, masked=True) -> torch.Tensor:
        """

        :param outputs:
        :param pl_module:
        :return: images: torch.Tensor of shape (B, C, H, W), C=3 [dense encoding] or C=4 [old encoding]
        """
        assert self.dataset_cfg.data_type == "graph", "Error in obtain_imgs(). This method is only valid for graph datasets."

        if self.dataset_cfg.encoding == "dense":
            Fx = pl_module.decoder.to_graph_features(logits=outputs["logits_heads"], probabilistic=False, masked=masked)
            if self.logging_cfg.rendering == "minigrid": #Not sure this should be the decider
                grids = tr.Nav2DTransforms.graph_features_to_minigrid(Fx)
                images = tr.Nav2DTransforms.minigrid_to_minigrid_render(grids, tile_size=self.logging_cfg.tile_size)
            else:
                raise NotImplementedError(f"Rendering method {self.logging_cfg.rendering} not implemented for dense graph encoding.")
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
            probs_heads = pl_module.decoder.param_p(outputs["logits_heads"])
            heatmap_layouts = {}
            for i, node in enumerate(pl_module.decoder.output_distributions.layout.attributes):
                heatmap_layouts[node] = self.prob_heatmap_fx(probs_heads['layout'][...,i], grid_dim)
        else:
            raise NotImplementedError(f"Encoding {self.dataset_cfg.encoding} not implemented.")

        heatmap_start = self.prob_heatmap_fx(probs_heads['start_location'].squeeze(-1), grid_dim)
        heatmap_goal = self.prob_heatmap_fx(probs_heads['goal_location'].squeeze(-1), grid_dim)

        self.log_images(trainer, tag + "/prob_heatmap/start", heatmap_start, mode="RGBA")
        self.log_images(trainer, tag + "/prob_heatmap/goal", heatmap_goal, mode="RGBA")
        for node, heatmap_layout in heatmap_layouts.items():
            self.log_images(trainer, tag + f"/prob_heatmap/{node}", heatmap_layout, mode="RGBA")

    def prob_heatmap_fx(self, probs_fx, grid_dim):
        probs_fx = probs_fx.squeeze(0)
        assert len(probs_fx.shape) == 2

        probs_fx = probs_fx / probs_fx.amax(dim=[i for i in range(1, len(probs_fx.shape))], keepdim=True)
        probs_fx = probs_fx.reshape(probs_fx.shape[0], grid_dim, grid_dim)
        heat_map = torch.zeros((*probs_fx.shape, 4)) # (B, H, W, C), C=RGBA
        heat_map[..., 0] = 1  # all values will be shades of red
        heat_map[..., 3] = probs_fx
        heat_map = einops.rearrange(heat_map, 'b h w c -> b c h w')

        heat_map = self.node_to_gw_mapping(heat_map) #add zeros in between for inactive nodes

        return heat_map

    def normalise_metrics(self, metrics, device=torch.device("cpu")):

        for key in metrics.keys():
            if isinstance(metrics[key], list):
                metrics[key] = torch.tensor(metrics[key])
            if key in self.label_descriptors_config.keys():
                metrics[key] = metrics[key].to(device)
                metrics[key] /= self.label_descriptors_config[key].normalisation_factor
                # if isinstance(metrics[key], torch.Tensor):
                #     metrics[key] = metrics[key].tolist()

    def prob_moss_over_spl(self, graphs:List[nx.Graph], targets=None) -> Dict[int, float]:

        spl_moss = defaultdict(int)
        spl_nav = defaultdict(int)

        for m, g_nx in enumerate(graphs):
            if targets is None:
                goal_node = [n for n in g_nx.nodes if g_nx.nodes[n]['goal'] == 1.0]
                assert len(goal_node) == 1
                goal_node = goal_node[0]
                spl_graph = dict(nx.single_target_shortest_path_length(g_nx, goal_node))
            else:
                spl_graph = self.label_contents['shortest_path_dist'][targets[m].item()]
            for n, s in spl_graph.items():
                if g_nx.nodes[n]['goal'] != 1.0 and (g_nx.nodes[n]['moss'] == 1.0 or g_nx.nodes[n]['empty'] == 1.0):
                    spl_nav[s] += 1
                    if g_nx.nodes[n]['moss'] == 1.0:
                        assert g_nx.nodes[n]['empty'] != 1.0
                        spl_moss[s] += 1

        spl_nav = {k: spl_nav[k] for k in sorted(spl_nav.keys())}
        spl_moss = {k: spl_moss[k] for k in sorted(spl_nav.keys())}
        moss_vs_spl_p = {k: spl_moss[k]/spl_nav[k] for k in spl_nav.keys()}
        return moss_vs_spl_p, spl_nav, spl_moss

    def prob_lava_over_spl(self, graphs:List[nx.Graph], targets=None) -> Dict[int, float]:

        grid_size = tuple([int(np.sqrt(self.dataset_cfg.max_nodes))] * 2)
        depth = self.dataset_metadata.config.lava_distribution_params.get("sampling_depth", 3)

        spl_lava = defaultdict(int)
        spl_non_nav = defaultdict(int)

        for m, g_nx in enumerate(graphs):
            if targets is None:
                goal_node = [n for n in g_nx.nodes if g_nx.nodes[n]['goal'] == 1.0]
                assert len(goal_node) == 1
                goal_node = goal_node[0]
                spl_nav_graph = dict(nx.single_target_shortest_path_length(g_nx, goal_node))
            else:
                spl_nav_graph = self.label_contents['shortest_path_dist'][targets[m].item()]
            non_nav_nodes = [n for n in g_nx.nodes if g_nx.nodes[n]['wall'] == 1.0 or g_nx.nodes[n]['lava'] == 1.0]
            spl_non_nav_graph = get_non_nav_spl(non_nav_nodes, spl_nav_graph, grid_size, depth)
            for n, s in spl_non_nav_graph.items():
                if g_nx.nodes[n]['lava'] == 1.0 or g_nx.nodes[n]['wall'] == 1.0:
                    spl_non_nav[s] += 1
                    if g_nx.nodes[n]['lava'] == 1.0:
                        assert g_nx.nodes[n]['wall'] != 1.0
                        spl_lava[s] += 1

        spl_non_nav = {k: spl_non_nav[k] for k in sorted(spl_non_nav.keys())}
        spl_lava = {k: spl_lava[k] for k in sorted(spl_non_nav.keys())}
        lava_vs_spl_p = {k: spl_lava[k]/spl_non_nav[k] for k in spl_non_nav.keys()}
        return lava_vs_spl_p, spl_non_nav, spl_lava

    def spl_start(self, graphs:List[nx.Graph], targets=None, normalise=None) -> Tuple[np.ndarray, np.ndarray]:

        spl = []
        spl_start = []
        for m, g in enumerate(graphs):
            goal_node_id = [n for n in g.nodes if g.nodes[n]['goal'] == 1.0]
            assert len(goal_node_id) == 1
            goal_node_id = goal_node_id[0]
            if targets is None:
                s = dict(nx.single_target_shortest_path_length(g, goal_node_id))
            else:
                s = self.label_contents['shortest_path_dist'][targets[m].item()]
            if goal_node_id in s.keys():
                s.pop(goal_node_id)
            spl.append(np.array(list(s.values())))
            start_node_id = [n for n in g.nodes if g.nodes[n]['start'] == 1.0]
            assert len(start_node_id) == 1
            start_node_id = start_node_id[0]
            spl_start.append(s[start_node_id])

        spl_start = np.array(spl_start)
        spl_med = np.array([np.median(s) for s in spl])
        # idea: look at histogram of abs(spl_start - spl_med)
        #from https://journalspress.com/LJRS_Volume19/600_Data-Normalization-using-Median-&-Median-Absolute-Deviation-(MMAD)-based-Z-Score-for-Robust-Predictions-vs-Min%E2%80%93Max-Normalization.pdf
        if normalise == "MMAP":
            spl_dev_med = np.array([scipy.stats.median_abs_deviation(s) for s in spl])
            spl = [(s - spl_med[m])/spl_dev_med[m] for m, s in enumerate(spl)]
            spl_start = (spl_start - spl_med)/spl_dev_med

        return spl_start, spl_med