import logging

import dgl
import networkx as nx
import numpy as np
from typing import List, Union, Tuple, Optional, Any
import torch
import pytorch_lightning as pl
import wandb
import einops
import pandas as pd
from typing import Dict

from data_generators import Batch, OBJECT_TO_CHANNEL_AND_IDX
encode_graph_to_gridworld = Batch.encode_graph_to_gridworld
encode_reduced_adj_to_gridworld_layout = Batch.encode_reduced_adj_to_gridworld_layout
encode_decoder_mode_to_graph = Batch.encode_decoder_mode_to_graph
from util.transforms import DilationTransform, FlipBinaryTransform
from util.graph_metrics import compute_metrics
from copy import deepcopy

logger = logging.getLogger(__name__)

class GraphVAELogger(pl.Callback):
    def __init__(self, label_contents:Dict, samples, attributes, used_attributes, num_samples=32, max_cached_batches=0, accelerator="cpu"):
        super().__init__()
        device = torch.device("cuda" if accelerator == "gpu" else "cpu")
        self.label_contents = label_contents
        self.num_samples = num_samples
        self.max_cached_batches = max_cached_batches
        self.attributes = attributes
        self.used_attributes = used_attributes
        self.graphs = {}
        self.labels = {}
        self.gw = {}
        self.imgs = {}
        self.node_to_gw_mapping = DilationTransform(1)
        self.validation_batch = {
            "graph": [],
            "label_ids": [],
        }
        self.validation_step_outputs = {
            "loss" : [],
            "logits_A": [],
            "logits_Fx": [],
            "mean": [],
            "var_unconstrained": [],
        }

        self.predict_batch = deepcopy(self.validation_batch)
        self.predict_step_outputs = deepcopy(self.validation_step_outputs)

        try:
            for key in ["train", "val", "test"]:
                self.graphs[key], self.labels[key] = samples[key]
                self.graphs[key] = dgl.unbatch(self.graphs[key])
                self.graphs[key] = self.graphs[key][:num_samples]
                self.labels[key] = self.labels[key][:num_samples]
                self.graphs[key] = dgl.batch(self.graphs[key]).to(device)
                self.labels[key] = self.labels[key].to(device)
        except KeyError:
            logger.info(f"{key} dataset was not supplied in the samples provided to the GraphVAELogger")

        for key in self.graphs.keys():
            self.gw[key] = self.encode_graph_to_gridworld(self.graphs[key], self.attributes).to(device)
            self.imgs[key] = self.gridworld_to_img(self.gw[key]).to(device)

    # Main logging logic

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        try:
            self.log_images(trainer, "examples/train", self.imgs["train"], self.labels["train"])
            self.log_images(trainer, "examples/val", self.imgs["val"], self.labels["val"])
            self.log_images(trainer, "examples/test", self.imgs["test"], self.labels["test"])
        except KeyError as e:
            logger.info(f"{e} dataset was not supplied in the samples provided to the GraphVAELogger")

    def on_validation_epoch_end(self, trainer, pl_module):
        if "train" in self.graphs.keys():
            elbos, logits_A, logits_Fx, mean, var_unconstrained = self.obtain_model_outputs(self.graphs["train"], pl_module)
            reconstructed_imgs_train = self.obtain_imgs(logits_A, logits_Fx, pl_module)

            self.log_images(trainer, "reconstructions/train", reconstructed_imgs_train, self.labels["train"])
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/train", logits_A=logits_A, logits_Fx=logits_Fx)

        if "val" in self.graphs.keys():
            elbos, logits_A, logits_Fx, mean, var_unconstrained = self.obtain_model_outputs(self.graphs["val"], pl_module)
            reconstructed_imgs_val = self.obtain_imgs(logits_A, logits_Fx, pl_module)
            self.log_images(trainer, "reconstructions/val", reconstructed_imgs_val, self.labels["val"])
            self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="reconstructions/val", logits_A=logits_A, logits_Fx=logits_Fx)

        prior_samples = torch.randn(mean.shape).to(device=pl_module.device)
        logits_A_prior, logits_Fx_prior = pl_module.decoder(prior_samples)
        generated_imgs = self.obtain_imgs(logits_A_prior, logits_Fx_prior, pl_module)
        self.log_images(trainer, "generated/prior", generated_imgs)
        self.log_prob_heatmaps(trainer=trainer, pl_module=pl_module, tag="generated/prior", logits_A=logits_A_prior, logits_Fx=logits_Fx_prior)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        GraphVAELogger.prepare_stored_batch(self.validation_batch, self.validation_step_outputs)
        self.log_latent_embeddings(trainer, pl_module, "latent_space/val", mode="val")

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        GraphVAELogger.prepare_stored_batch(self.predict_batch, self.predict_step_outputs)
        self.log_latent_embeddings(trainer, pl_module, "latent_space/train", mode="predict")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "test" in self.graphs.keys():
            elbos, logits_A, logits_Fx, mean, var_unconstrained = self.obtain_model_outputs(self.graphs["train"],
                                                                                            pl_module)
            reconstructed_imgs = self.obtain_imgs(logits_A, logits_Fx, pl_module)
            self.log_images(trainer, "reconstructions/test", reconstructed_imgs, self.labels["test"])

    # Boiler plate code

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.clear_stored_batches()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.clear_stored_batches()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.clear_stored_batches()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
            GraphVAELogger.store_batch(self.validation_batch, self.validation_step_outputs, batch, outputs)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Tuple,
        batch: Any,
        batch_idx: int,
        data_loader_idx: int = 0,
    ) -> None:
        if batch_idx < self.max_cached_batches:
            self.store_batch(self.predict_batch, self.predict_step_outputs, batch, outputs)

    # utility methods

    def clear_stored_batches(self):
        self.validation_batch = {
            "graph": [],
            "label_ids": [],
        }
        self.validation_step_outputs = {
            "loss" : [],
            "logits_A": [],
            "logits_Fx": [],
            "mean": [],
            "var_unconstrained": [],
        }
        self.predict_batch = deepcopy(self.validation_batch)
        self.predict_step_outputs = deepcopy(self.validation_step_outputs)

    def obtain_model_outputs(self, graphs, pl_module):

        elbos, logits_A, logits_Fx, mean, var_unconstrained = \
            pl_module.all_model_outputs_pathwise(graphs, num_samples=pl_module.hparams.config_logging.num_variational_samples)
        return elbos, logits_A, logits_Fx, mean, var_unconstrained

    @staticmethod
    def prepare_stored_batch(batch_dict, output_dict):

        def flatten(l):
            return [item for sublist in l for item in sublist]

        unbatched_graphs = [dgl.unbatch(g) for g in batch_dict["graph"]]
        batch_dict["graph"] = flatten(unbatched_graphs)
        batch_dict["label_ids"] = torch.cat(batch_dict["label_ids"])

        for key in output_dict.keys():
            output_dict[key] = torch.cat(output_dict[key])

    @staticmethod
    def store_batch(batch_dict, output_dict, batch, output):
        loss, logits_A, logits_Fx, mean, var_unconstrained = output

        batch_dict["graph"].append(batch[0])
        batch_dict["label_ids"].append(batch[1])

        output_dict["loss"].append(loss)
        output_dict["logits_A"].append(logits_A)
        output_dict["logits_Fx"].append(logits_Fx)
        output_dict["mean"].append(mean)
        output_dict["var_unconstrained"].append(var_unconstrained)

    def obtain_imgs(self, logits_A, logits_Fx, pl_module):

        mode_probs = pl_module.decoder.param_m((logits_A, logits_Fx))
        reconstructed_gws = self.encode_graph_to_gridworld(mode_probs, attributes=self.used_attributes)
        reconstructed_imgs = self.gridworld_to_img(reconstructed_gws)

        return reconstructed_imgs

    def log_images(self, trainer, tag, images, labels=None, mode=None):
        if labels is not None:
            captions = [f"Label:{y}" for y in labels]
        else:
            captions = [None] * len(images)
        trainer.logger.experiment.log({
            tag: [wandb.Image(x, caption=c, mode=mode)
                         for x, c in zip(images, captions)],
            "global_step": trainer.global_step
        })

    def log_prob_heatmaps(self, trainer, pl_module, tag, logits_A, logits_Fx):
        logits_A, logits_Fx = pl_module.decoder.param_p((logits_A, logits_Fx))
        grid_dim = int(np.sqrt(logits_Fx.shape[-2])) #sqrt(num_nodes)
        heatmap_start = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("start")], grid_dim)
        heatmap_goal = self.prob_heatmap_fx(logits_Fx[..., pl_module.decoder.attributes.index("goal")], grid_dim)
        heatmap_layout = self.prob_heatmap_A(logits_A, grid_dim)
        self.log_images(trainer, tag + "/prob_heatmap/start", heatmap_start, mode="RGBA")
        self.log_images(trainer, tag + "/prob_heatmap/goal", heatmap_goal, mode="RGBA")
        self.log_images(trainer, tag + "/prob_heatmap/layout", heatmap_layout, mode="RGBA")

    def log_latent_embeddings(self, trainer, pl_module, tag, mode="val"):
        if mode == "val":
            Z = self.validation_step_outputs["mean"]
            logits_A, logits_Fx = self.validation_step_outputs["logits_A"], self.validation_step_outputs["logits_Fx"]
            graphs = self.validation_batch["graph"]
            labels = self.validation_batch["label_ids"]
        elif mode == "predict":
            Z = self.predict_step_outputs["mean"]
            logits_A, logits_Fx = self.predict_step_outputs["logits_A"], self.predict_step_outputs["logits_Fx"]
            graphs = self.predict_batch["graph"]
            labels = self.predict_batch["label_ids"]

        assert Z.shape[0] == logits_A.shape[0] == logits_Fx.shape[0]

        Z = Z.cpu().numpy()

        mode_probs_A, mode_probs_Fx = pl_module.decoder.param_m((logits_A, logits_Fx))
        reconstructed_graphs = encode_decoder_mode_to_graph(mode_probs_A, mode_probs_Fx, make_valid=False,
                                                            device=torch.device("cpu"))
        reconstruction_metrics = compute_metrics(
            reconstructed_graphs,
            desired_metrics=["valid","solvable","shortest_path", "resistance", "navigable_nodes"],
            start_dim=pl_module.decoder.attributes.index("start"),
            goal_dim=pl_module.decoder.attributes.index("goal"))

        reconstructed_imgs = self.obtain_imgs(logits_A, logits_Fx, pl_module)

        del logits_A, logits_Fx, mode_probs_A, mode_probs_Fx, reconstructed_graphs

        input_gws = self.encode_graph_to_gridworld(graphs, self.attributes)
        input_imgs = self.gridworld_to_img(input_gws)
        del input_gws

        col_names = ["Z"+str(i) for i in range(Z.shape[-1])]
        df = pd.DataFrame(Z, columns=col_names)
        # Store pre-computed Input metrics
        for input_property in reversed(["task_structure", "shortest_path", "resistance", "navigable_nodes", "seed"]):
            if isinstance(self.label_contents[input_property], list):
                data = [self.label_contents[input_property][i] for i in labels]
            else:
                data = self.label_contents[input_property][labels]
            df.insert(0, f"Input_{input_property}", data)

        # Store Reconstruction metrics
        for key in reconstruction_metrics.keys():
            df.insert(0, f"Reconstruction_{key}", reconstruction_metrics[key])
        del reconstruction_metrics

        df.insert(0, "Inputs", [wandb.Image(x, mode="RGBA") for x in input_imgs])
        df.insert(0, "Reconstructions", [wandb.Image(x, mode="RGBA") for x in reconstructed_imgs])
        df.insert(0, "Label_ids", labels.tolist())

        trainer.logger.experiment.log({
            tag: wandb.Table(
                dataframe=df
            ),
            "global_step": trainer.global_step
        })

    def prob_heatmap_fx(self, probs_fx, grid_dim):
        #TODO: at some point good to look into using those directly
        colors = {
            "bright_red": [1, 0, 0, 1],
            "light_red": [1, 0, 0, 0.1],
            "bright_blue": [0, 0, 1, 1],
            "light_blue": [0, 0, 1, 0.1],
            "white": [0, 0, 0, 0],
            "black": [0, 0, 0, 1],
        }

        assert len(probs_fx.shape) == 2

        probs_fx = probs_fx / probs_fx.amax(dim=[i for i in range(1, len(probs_fx.shape))], keepdim=True)
        probs_fx = probs_fx.reshape(probs_fx.shape[0], grid_dim, grid_dim)
        heat_map = torch.zeros((*probs_fx.shape, 4)) # (B, H, W, C), C=RGBA
        heat_map[..., 0] = 1  # all values will be shades of red #TODO: could do color according to start/goal config
        heat_map[..., 3] = probs_fx
        heat_map = einops.rearrange(heat_map, 'b h w c -> b c h w')

        heat_map = self.node_to_gw_mapping(heat_map) #add zeros in between for inactive nodes

        return heat_map

    def prob_heatmap_A(self, probs_A, grid_dim):

        assert len(probs_A.shape) == 2 #check A is flattened

        # TODO: to revise for non reduced formulations
        probs_A = probs_A.reshape(probs_A.shape[0], -1, 2)

        flip_bits = FlipBinaryTransform()
        threshold = .5
        layout_dim = int(grid_dim * 2 + 1)
        layout_gw = encode_reduced_adj_to_gridworld_layout(probs_A, (layout_dim, layout_dim), probalistic_mode=True, prob_threshold=threshold)

        heat_map = torch.zeros((*layout_gw.shape, 4)).to(layout_gw)  # (B, H, W, C), C=RGBA
        heat_map[..., 0][layout_gw >= threshold] = 1 #val >= threshold go to red channel
        heat_map[..., 3][layout_gw >= threshold] = (layout_gw[layout_gw >= threshold] - threshold)*2 #set transparency according to prob rescaled from [treshold,1] to [0,1]
        heat_map[..., 2][layout_gw < threshold] = 1 #val <= threshold go to red channel
        heat_map[..., 3][layout_gw < threshold] = flip_bits((layout_gw[layout_gw < threshold])*2)  #set transparency according to prob rescaled from [0,treshold] to [0,1], then flipped to [1,0]
        heat_map = einops.rearrange(heat_map, 'b h w c -> b c h w')

        return heat_map


    def encode_graph_to_gridworld(self, graphs, attributes):
        return encode_graph_to_gridworld(graphs, attributes, self.used_attributes)

    def gridworld_to_img(self, gws):
        colors = {
            'wall': [0., 0., 0., 1.],   #black
            'empty': [0., 0., 0., 0.],  #white
            'start': [0., 0., 1., 1.],  #blue
            'goal': [0., 1., 0., 1.],   #green
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
