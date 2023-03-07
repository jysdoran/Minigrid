__package__ = "maze_representations"

from argparse import Namespace

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from einops import rearrange
import networkx as nx

import gym_minigrid.minigrid as minigrid
from data_loaders import GridNavDataModule

from .models.graphVAE import LightningGraphVAE
from .util import graph_metrics
from .util import transforms as tr
from .util.util import cdist_polar, interpolate_between_pairs


# Workaround so I don't have to deal with the argparser here...
def dict_to_namespace(d):
    x = Namespace()
    _ = [setattr(x, k, dict_to_namespace(v)) 
         if isinstance(v, dict) 
         else setattr(x, k, v) 
         for k, v in d.items() ]
    return x

args = {
    'generative_model_checkpoint_path': './maze_representations/checkpoints/graphvae_densegraph_22k_icml23.ckpt',
    'dataset_path': './maze_representations/datasets/minigrid_dense_graph_22k',
    'generative_model_batch_size': 128,
}

args = dict_to_namespace(args)

interpolation_config = {
    'num_samples': 3,
    'num_interpolations': 5,
    # 'remove_noninformative_dimensions': False,
    # 'dimension_reduction_threshold': 0.9,
    'sample_latents': False,
    'interpolation_scheme': 'polar',
    
    'tile_size': 1,
    'rendering_method': 'minigrid',
}
interpolation_config = dict_to_namespace(interpolation_config)


data_module = GridNavDataModule(args.dataset_path,
                                batch_size=args.generative_model_batch_size,
                                num_samples=1,
                                num_workers=0,
                                val_data='test')

data_module.setup()
train_dataset = data_module.train
val_dataset = data_module.val
test_dataset = data_module.test

generative_model = LightningGraphVAE.load_from_checkpoint(
                    args.generative_model_checkpoint_path).cpu()

# Adapted from logger_callbacks.py
def obtain_imgs_and_grids(outputs, 
                pl_module:pl.LightningModule, 
                rendering_method: str,
                tile_size,
                masked=True,
                ) -> torch.Tensor:
        """

        :param outputs:
        :param pl_module:
        :return: images: torch.Tensor of shape (B, C, H, W), C=3 [dense encoding] or C=4 [old encoding]
        """
        # assert self.dataset_cfg.data_type == "graph", "Error in obtain_imgs(). This method is only valid for graph datasets."

        if pl_module.decoder.shared_params.data_encoding == "dense":
            Fx = pl_module.decoder.to_graph_features(outputs["logits_heads"], masked=masked, probabilistic=False)
            if rendering_method == "minigrid": #Not sure this should be the decider
                grids = tr.Nav2DTransforms.graph_features_to_minigrid(Fx)
                images = tr.Nav2DTransforms.minigrid_to_minigrid_render(grids, tile_size=tile_size)
            else:
                raise NotImplementedError(f"Rendering method {rendering_method} not implemented for dense graph encoding.")
        # elif pl_module.decoder.shared_params.data_encoding == "minimal":
        #     assert rendering_method != "minigrid", "Error in obtain_imgs(). Minigrid rendering is not implemented for minimal graph encoding."
        #     print(f"Warning {rendering_method} is Deprecated. Correct behavior not guaranteed.")
        #     mode_probs = pl_module.decoder.param_m((outputs["logits_A"], outputs["logits_Fx"]))
        #     reconstructed_gws = self.encode_graph_to_gridworld(mode_probs, attributes=pl_module.decoder.attributes)
        #     images = self.gridworld_to_img(reconstructed_gws)
        else:
            raise NotImplementedError(f"Encoding {pl_module.decoder.shared_params.data_encoding} not implemented.")

        return images, grids

# Adapted from logger_callbacks.py
def get_latent_interpolation(pl_module:pl.LightningModule,
                             init_datapoints,
                             end_datapoints,
                             interpolations_per_samples:int=8,
                            #  remove_dims:bool=True,
                            #  dim_reduction_threshold:float=0.9,
                             interpolation_scheme:str= 'linear',
                             latent_sampling:bool=True,
                             *,
                             tile_size,
                             rendering_method):
    
    init_datapoints_batch = dgl.batch(init_datapoints)
    end_datapoints_batch = dgl.batch(end_datapoints)
    with torch.inference_mode():
        init_mean, init_std = pl_module.encoder(init_datapoints_batch)
        end_mean, end_std = pl_module.encoder(end_datapoints_batch)
    mean = torch.concat([init_mean, end_mean], dim=0)
    std = torch.concat([init_std, end_std], dim=0)
    
    pairs = torch.stack([torch.arange(len(init_datapoints)),  len(init_datapoints)+torch.arange(len(end_datapoints))], dim=1)

    # # Remove dimensions with average standard deviation of > dim_reduction_threshold (e.g. uninformative)
    # if remove_dims and mode!="prior":
    #     kept_dims = torch.where(std < dim_reduction_threshold)[-1]
    #     kept_dims = torch.unique(kept_dims)
    #     mean = mean[..., kept_dims]
    #     std = std[..., kept_dims]

    # sort the distances in ascending order
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

    # # reassembly
    # if remove_dims and mode!="prior":
    #     temp = interp_Z.clone().to(device=pl_module.device)
    #     interp_Z = torch.zeros(*interp_Z.shape[:-1], Z_dim,
    #                                                 dtype=torch.float).to(pl_module.device)
    #     interp_Z[..., kept_dims] = temp
    #     del temp

    all_Z = interp_Z
    del mean, std, interp_Z

    logits = {}
    if pl_module.decoder.shared_params.data_encoding == "minimal":
        logits["logits_A"], logits["logits_Fx"] = pl_module.decoder(all_Z)
    elif pl_module.decoder.shared_params.data_encoding == "dense":
        logits["logits_Fx"] = pl_module.decoder(all_Z)
    else:
        raise NotImplementedError(f"Encoding {pl_module.shared_params.data_encoding} not implemented.")
    del all_Z
    
    init_imgs = tr.Nav2DTransforms.dense_graph_to_minigrid_render(init_datapoints, tile_size=tile_size)
    end_imgs = tr.Nav2DTransforms.dense_graph_to_minigrid_render(end_datapoints, tile_size=tile_size)
    imgs, grids = obtain_imgs_and_grids(logits, pl_module,
                                        rendering_method=rendering_method,
                                        tile_size=tile_size)
    
    return {
        'init_imgs': init_imgs,
        'end_imgs': end_imgs,
        'interp_imgs': imgs,
        'interp_grids': grids,
    }


# NOTE: You should change this to use some environments that are good/bad for the agent
init_datapoints, init_idxs = train_dataset[:interpolation_config.num_samples]
end_datapoints, end_idxs  = train_dataset[interpolation_config.num_samples:interpolation_config.num_samples*2]


interp_out = get_latent_interpolation(pl_module=generative_model,
                                      init_datapoints=init_datapoints,
                                      end_datapoints=end_datapoints,
                                      interpolations_per_samples=interpolation_config.num_interpolations,
                                      # remove_dims=interpolation_config.remove_noninformative_dimensions,
                                      # dim_reduction_threshold=interpolation_config.dimension_reduction_threshold,
                                      latent_sampling=interpolation_config.sample_latents,
                                      interpolation_scheme=interpolation_config.interpolation_scheme,
                                      tile_size=interpolation_config.tile_size,
                                      rendering_method=interpolation_config.rendering_method)

init_imgs = interp_out['init_imgs']
end_imgs = interp_out['end_imgs']
interp_imgs = interp_out['interp_imgs']
interp_grids = interp_out['interp_grids']

init_grids = tr.Nav2DTransforms.dense_graph_to_minigrid(init_datapoints)
end_grids = tr.Nav2DTransforms.dense_graph_to_minigrid(end_datapoints)

# Obtain the interpolated samples + original pairs
all_grids = []
interp_grids = rearrange(interp_grids, '(p b) ... -> p b ...', p=len(init_datapoints))
for i in range(len(init_datapoints)):
    all_grids.append(init_grids[i])
    for img in interp_grids[i]:
        all_grids.append(img)
    all_grids.append(end_grids[i])
grids = np.stack(all_grids)

# TODO Replace with agents regret
def compute_resistance_for_grids(grids):
    init_metrics = []
    for grid, nx_grid in zip(grids, tr.Nav2DTransforms.minigrid_to_dense_graph(grids)):
        start = np.argwhere(grid[..., 0] == minigrid.OBJECT_TO_IDX['agent'])[0]
        goal = np.argwhere(grid[..., 0] == minigrid.OBJECT_TO_IDX['goal'])[0]
        start = tuple(start-1)
        goal = tuple(goal-1)
        # metric = graph_metrics.is_solvable(nx_grid, source=start, target=goal)
        nx_grid, valid, solvable = graph_metrics.prepare_graph(nx_grid, start, goal)
        if valid and solvable:
            metric = graph_metrics.resistance_distance(nx_grid, source=start, target=goal)
        else:
            metric = float('nan')
        init_metrics.append(metric)
        
    return np.array(init_metrics)
    
metrics = compute_resistance_for_grids(grids)

#
# Plotting
#

# Obtain the interpolated samples + original pairs
all_imgs = []
imgs = rearrange(interp_imgs, '(p b) ... -> p b ...', p=len(init_datapoints))
for i in range(len(init_datapoints)):
    all_imgs.append(init_imgs[i])
    for img in imgs[i]:
        all_imgs.append(img)
    all_imgs.append(end_imgs[i])
imgs = torch.stack(all_imgs)

img_per_row = interpolation_config.num_interpolations+2

fig, axes = plt.subplots(len(init_datapoints)*2, 1, figsize=(7, 6.7))
for i in range(len(init_datapoints)):
    metric = metrics[i*img_per_row: (i+1)*img_per_row]
    im_width = 1. / img_per_row
    x = np.arange(start=im_width/2, stop=1-im_width/2, step=im_width)
    axes[i*2].plot(x, metric, marker='x')
    axes[i*2].set_xlim(0, 1)
    axes[i*2].grid()
    axes[i*2].set(xticklabels=[], xticks=np.arange(1, step=im_width))
    for tic in axes[i*2].xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    # TODO: replace with metric name
    axes[i*2].set_ylabel('metric')
    
    imgs_i = imgs[i*img_per_row: (i+1)*img_per_row]
    img_grid = torchvision.utils.make_grid(imgs_i, nrow=img_per_row, normalize=True, padding=1)
    img_grid = torchvision.transforms.functional.to_pil_image(img_grid)
    axes[i*2+1].imshow(np.asarray(img_grid))
    axes[i*2+1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

fig.tight_layout()
fig.show()
breakpoint()

fig.savefig('./interpolations_vs_metric.pdf')
