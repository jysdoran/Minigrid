import itertools
from pathlib import Path
import sys
import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fileinput

from data_loaders import Maze_Dataset
from util import *
from models.VAE import *
from torch.utils.tensorboard import SummaryWriter

#Select the directory using this
dataset_size = 120000
task_structures = ('rooms_unstructured_layout', 'maze') #{'maze', 'rooms_unstructured_layout'}
data_type = 'grid'
data_dim = 27

use_gpu = False
embedding = False
batch_size = 64
epochs = 100
arch = 'fc'
latent_dim = 12

task_structures = '-'.join(task_structures)
dataset_directory = f"ts={task_structures}-x={data_type}-s={dataset_size}-d={data_dim}"
run_name = f"{dataset_directory}_arch={arch}-z={latent_dim}_b={batch_size}-e={epochs}"# 'test'#'multiroom10000x27_6cnn_z12_b64e1000' #CHANGE

writer = SummaryWriter('runs/' + run_name)
base_dir = str(Path(__file__).resolve().parent)
datasets_dir = base_dir + '/datasets/'

cifar_dir = datasets_dir + 'cifar10_data'
mnist_dir = datasets_dir #+ 'MNIST'

nav_dir = datasets_dir + dataset_directory

if data_type == 'grid':
    layout_channels = (0,1)
elif data_type == 'gridworld':
    layout_channels = 0

model_names = [
#    'VAE_cnn_maze10000x27_cnn_fc',
#     'VAE_cnn_maze10000x27_fc_sd',
#     'VAE_cnn_maze10000x27_fc_l2',
#     'VAE_cnn_maze10000x27_l3_s',
#     'VAE_cnn_maze10000x27_l3',
#     'VAE_maze10000x27_6cnn_fc_bs50_e250',
#     'VAE_maze10000x27_enc_6cnn_fc_dec_fc_2dconv_bs50_e250',
#     'VAE_maze10000x27_6cnn_fc_bs50_e10',
#     'multiroom10000x27_6cnn_z32_b64e2000',
#    'multiroom10000x27_6cnn_z12_b64e2000',
    run_name,
#    'test',
]

if data_type == 'grid':
    layout_channels = (0,1)
elif data_type == 'gridworld':
    layout_channels = (0,)

# #Uncomment
train_data = Maze_Dataset(
          nav_dir, train=True,
          transform = transforms.Compose([
              SelectChannelsTransform(*layout_channels),
              transforms.ToTensor(),
          ]))

# train_data = torchvision.datasets.MNIST(
#     mnist_dir, train=True, download=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         FlattenTransform(1, -1),
#         BinaryTransform(0.6),
#         ]))
# test_data = torchvision.datasets.MNIST(
#     mnist_dir, train=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         FlattenTransform(1, -1),
#         BinaryTransform(0.6),
#         ]))

# Specify the hyperpameter choices

for model_name in model_names:
    tensorboard = SummaryWriter('runs/' + model_name)
    # address bug overwritting projector_config when a new embedding is created
    if embedding:
        projector_config_file = Path.cwd() / tensorboard.log_dir / 'projector_config.pbtxt'
        projector_config_file.rename(str(projector_config_file)+'.bkup')

    if not use_gpu: device = torch.device('cpu')
    else: device = None
    model, optimiser, args = load_state('checkpoints/'+model_name+'.pt', model_type=VAE, optim_type=optim.Adam, device=device)
    args.cuda = use_gpu
    args.device = torch.device("cuda" if args.cuda else "cpu")
    bin_threshold = 0.5
    binary_transform = BinaryTransform(bin_threshold)
    flip_bits = FlipBinaryTransform()
    resize = torchvision.transforms.Resize((100, 100), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    train_loader = create_dataloader(train_data, args)
    example_data_train, example_targets_train = next(iter(train_loader))  # TODO: make deterministic accross runs

    with torch.inference_mode():
        #######

        model.eval()
        model = model.to(args.device)
        # mean, logvar = model.encoder(example_data_train)
        # latent_space_vis = plot_latent_visualisation(model, Z_points=mean, device=args.device, convert_from=train_data.dataset_metadata['data_type'])
        # tensorboard.add_figure('Latent space visualisation, Z ~ q(z|x), x ~ train data', latent_space_vis)
        # tensorboard.close()
        # sys.exit()

        # Compute latent space statistics
        # We don't use labels hence discard them with a _
        all_means = []
        all_stds = []
        for batch_idx, (mbatch, _) in enumerate(train_loader):
            mean, logvar = model.encoder(mbatch)
            all_means.append(mean)
            all_stds.append(torch.exp(logvar/2))

        all_means = torch.cat(all_means)
        all_stds = torch.cat(all_stds)

        variational_mean = all_means.mean(axis=0)
        variational_std = all_stds.mean(axis=0)

        prior_impossible_f = []
        polar_impossible_f = []
        for i in range(5):
            if isinstance(model.decoder.bottleneck.input_size, int):
                latent_dim = (model.decoder.bottleneck.input_size,)
            else:
                latent_dim = model.decoder.bottleneck.input_size
            Z = torch.randn(50, *latent_dim).to(args.device)  # M, B, D
            logits = model.decoder(Z)
            generated_data = model.decoder.param_b(logits)
            if train_data.dataset_metadata['data_type'] == 'grid':
                generated_samples = grid_to_gridworld(generated_data, layout_only=True)
            elif train_data.dataset_metadata['data_type'] == 'gridworld':
                generated_samples = generated_data
            else:
                raise RuntimeError(f"Data type {train_data.dataset_metadata['data_type']} not recognised.")

            node_groups = []
            impossible_layouts = []
            for layout in generated_samples:
                layout = layout.squeeze().detach().cpu().numpy()
                node_groups.append(numIslands(layout))
                if node_groups[-1] == 1:
                    impossible_layouts.append(False)
                else:
                    impossible_layouts.append(True)

            fraction_impossible = sum(impossible_layouts) / len(impossible_layouts)
            prior_impossible_f.append(fraction_impossible)

            example_data_train, example_targets_train = next(iter(train_loader))

            Z_interp, X_interp = latent_interpolation(model=model, minibatch=example_data_train,
                                                      Z_mean=variational_mean, Z_std=variational_std, labels=example_targets_train,
                                                      dim_r_threshold=.9, n_interp=16, interpolation_scheme='polar',
                                                      latent_sampling = True, device=args.device)
            X_interp = X_interp.reshape(X_interp.shape[0] * X_interp.shape[1], *X_interp.shape[2:])
            if train_data.dataset_metadata['data_type'] == 'grid':
                X_interp_gridworld = grid_to_gridworld(X_interp, layout_only=True)
            elif train_data.dataset_metadata['data_type'] == 'gridworld':
                X_interp_gridworld = X_interp
            else:
                raise RuntimeError(f"Data type {train_data.dataset_metadata['data_type']} not recognised.")

            node_groups = []
            impossible_layouts = []
            for layout in X_interp_gridworld:
                layout = layout.squeeze().detach().cpu().numpy()
                node_groups.append(numIslands(layout))
                if node_groups[-1] == 1:
                    impossible_layouts.append(False)
                else:
                    impossible_layouts.append(True)

            fraction_impossible = sum(impossible_layouts) / len(impossible_layouts)
            polar_impossible_f.append(fraction_impossible)

        prior_impossible_f = np.array(prior_impossible_f)
        polar_impossible_f = np.array(polar_impossible_f)
        tensorboard.add_text(f'Fraction of impossible layout from Prior sampling: (Mean, Mean standard error). '
                             f'Number of experiments: {np.size(prior_impossible_f)}', f'{prior_impossible_f.mean()} +/-'
                             f' {prior_impossible_f.std(ddof=1) / np.sqrt(np.size(prior_impossible_f))}')
        tensorboard.add_text(f'Fraction of impossible layout from noisy (1std) Polar interpolation: (Mean, Mean standard error). '
                             f'Number of experiments: {np.size(polar_impossible_f)}', f'{polar_impossible_f.mean()} +/-'
                             f' {polar_impossible_f.std(ddof=1) / np.sqrt(np.size(polar_impossible_f))}')

        # impossible_layouts = np.array(impossible_layouts).reshape(*Z_interp.shape[0:2])
        # ind = [str(i) for i in range(impossible_layouts.shape[-1])]
        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])
        # distribution = impossible_layouts.sum(axis=0)
        # ax.bar(ind, distribution)
        # tensorboard.add_figure('Distribution of impossible layouts along polar interpolation.', fig)

        img_grid = torchvision.utils.make_grid(flip_bits(resize(X_interp_gridworld)), nrow=Z_interp.shape[-2])
        #TODO store dimensionality of p(z|x) in TB
        tensorboard.add_image('Interpolated samples (noisy sampling), Polar', img_grid)

        if embedding:
            # address bug overwritting projector_config when a new embedding is created
            with open(str(projector_config_file) + '.temp', 'w') as fout, fileinput.input([str(projector_config_file) + '.bkup', str(projector_config_file)]) as fin:
                for line in fin:
                    fout.write(line)
            # cleanup files
            new_projector_config_file = Path(str(projector_config_file) + '.temp').rename(str(projector_config_file))
            Path(str(projector_config_file)+'.bkup').unlink()

        # node_groups = []
        # impossible_layouts = []
        # for layout in X_interp_gridworld:
        #     layout = layout.squeeze().detach().cpu().numpy()
        #     node_groups.append(numIslands(layout))
        #     if node_groups[-1] == 1:
        #         impossible_layouts.append(False)
        #     else:
        #         impossible_layouts.append(True)
        #
        # fraction_impossible = sum(impossible_layouts) / len(impossible_layouts)
        # tensorboard.add_text('Fraction of impossible layout for Interpolated samples, Polar: ', str(fraction_impossible))

        tensorboard.close()
        sys.exit()

print("Done")