import logging

import dgl
import numpy as np
from typing import List, Union, Tuple
import torch
import pytorch_lightning as pl
import wandb
import einops

from data_generators import Batch, OBJECT_TO_CHANNEL_AND_IDX
encode_graph_to_gridworld = Batch.encode_graph_to_gridworld
encode_reduced_adj_to_gridworld_layout = Batch.encode_reduced_adj_to_gridworld_layout
from util.transforms import DilationTransform, FlipBinaryTransform

logger = logging.getLogger(__name__)

class ImageLogger(pl.Callback):
    def __init__(self, samples, attributes, used_attributes, num_samples=32, accelerator="cpu"):
        super().__init__()
        device = torch.device("cuda" if accelerator == "gpu" else "cpu")
        self.num_samples = num_samples
        self.attributes = attributes
        self.used_attributes = used_attributes
        self.graphs = {}
        self.labels = {}
        self.gw = {}
        self.imgs = {}
        self.node_to_gw_mapping = DilationTransform(1)

        try:
            for key in ["train", "val", "test"]:
                self.graphs[key], self.labels[key] = samples[key]
                self.graphs[key] = dgl.unbatch(self.graphs[key])
                self.graphs[key] = self.graphs[key][:num_samples]
                self.labels[key] = self.labels[key][:num_samples]
                self.graphs[key] = dgl.batch(self.graphs[key]).to(device)
                self.labels[key] = self.labels[key].to(device)
        except KeyError:
            logger.info(f"{key} dataset was not supplied in the samples provided to the ImageLogger")

        for key in self.graphs.keys():
            self.gw[key] = self.encode_graph_to_gridworld(self.graphs[key], self.attributes).to(device)
            self.imgs[key] = self.gridworld_to_img(self.gw[key]).to(device)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # colors = {
        #     "bright_red": [1, 0, 0, 1],
        #     "light_red": [1, 0, 0, 0.1],
        #     "bright_blue": [0, 0, 1, 1],
        #     "light_blue": [0, 0, 1, 0.1],
        #     "white": [0, 0, 0, 0],
        #     "black": [0, 0, 0, 1],
        # }
        #
        # imgs = []
        # for val in colors.values():
        #     imgs.append(einops.repeat(torch.tensor(val), 'c->c h w', h=4, w=4))
        # imgs = torch.stack(imgs)
        #
        # self.log_images(trainer, "test/colors", imgs, colors.keys(), mode="RGBA")

        try:
            self.log_images(trainer, "examples/train", self.imgs["train"], self.labels["train"])
            self.log_images(trainer, "examples/val", self.imgs["val"], self.labels["val"])
            self.log_images(trainer, "examples/test", self.imgs["test"], self.labels["test"])
        except KeyError as e:
            logger.info(f"{e} dataset was not supplied in the samples provided to the ImageLogger")

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

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "test" in self.graphs.keys():
            elbos, logits_A, logits_Fx, mean, var_unconstrained = self.obtain_model_outputs(self.graphs["train"],
                                                                                            pl_module)
            reconstructed_imgs = self.obtain_imgs(logits_A, logits_Fx, pl_module)
            self.log_images(trainer, "reconstructions/test", reconstructed_imgs, self.labels["test"])

    def obtain_model_outputs(self, graphs, pl_module):

        elbos, logits_A, logits_Fx, mean, var_unconstrained = \
            pl_module.all_model_outputs_pathwise(graphs, num_samples=pl_module.hparams.config_logging.num_variational_samples)
        return elbos, logits_A, logits_Fx, mean, var_unconstrained

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

    def encode_graph_to_gridworld(self, graphs, attributes):
        return encode_graph_to_gridworld(graphs, attributes, self.used_attributes)

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

        #probs_fx = torch.sigmoid(probs_fx) # get to [0, 1] domain
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
