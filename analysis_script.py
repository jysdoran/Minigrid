import itertools
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_loaders import Maze_Dataset
from util import *
from models.VAE import *

base_dir = str(Path(__file__).resolve().parent)
dataset_dir = base_dir + '/datasets/'
cifar_dir = dataset_dir + 'cifar10_data'
maze_dir = dataset_dir + 'only_grid_10000x11'#'only_grid_10000x11'

img_size = 11

train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              #transforms.Resize((img_size, img_size)),
              transforms.ToTensor(),
          ]))

parser = create_base_argparser()
parser = VAE.add_model_args(parser)
parser.print_help()

# Specify the hyperpameter choices
data_dim = train_data[0][0].numel()
data_dims = (img_size,img_size,1)
args_dist_b = ['--dec_layer_dims', '2', '100', f'{data_dim}',
             '--enc_layer_dims', f'{data_dim}', '100', '2',
             '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--epochs', '50',
             '--learning_rate', '1e-4',
             '--cuda']

args_dist_b = parser.parse_args(args_dist_b)
args_dist_b.cuda = args_dist_b.cuda and torch.cuda.is_available()
args_dist_b.device = torch.device("cuda" if args_dist_b.cuda else "cpu")

# Seed all random number generators for reproducibility of the runs
seed_everything(args_dist_b.seed)

# Initialise the model and the Adam (SGD) optimiser
model_dist_b = VAE(args_dist_b).to(args_dist_b.device)
optimizer_dist_b = optim.Adam(model_dist_b.parameters(), lr=args_dist_b.learning_rate)

load_state(model_dist_b, optimizer_dist_b, 'checkpoints/VAE_discrete_onlygrid_10.pt')

N = 1000
Z = torch.randn(N, 2, device=args_dist_b.device)
bin_threshold = 0.5
binary_transform = BinaryTransform(bin_threshold)

with torch.inference_mode():
    # # TODO: put in a function
    # # Experiment: generating samples using standard Gaussian
    # # Sample the latents
    # logits = model_dist_b.decoder(Z)
    #
    # # Use ancestral sampling to produce samples
    # indices = np.random.choice(len(train_data), size=42)
    # original_samples = torch.vstack([train_data[idx][0] for idx in indices]).squeeze()
    # samples = model_dist_b.decoder.sample(logits, num_samples=1).detach().cpu()[0]
    # means = model_dist_b.decoder.mean(logits).detach().cpu()
    # # means = binary_transform(means)
    # params = binary_transform(model_dist_b.decoder.param_p(logits).detach().cpu())
    #
    # # Create a matrix of samples
    # X = torch.vstack([original_samples, samples, means, params])
    # X = X.view(-1, *data_dims).detach().cpu()
    # X = X.squeeze()
    #
    # # Create a figure
    # fig = plot_grid_of_samples(X, grid=(12, 14), figsize=(14, 10))
    # fig.suptitle('VAE with Bernoulli Variational Distribution, trained on 10 000 11x11 Mazes.', fontsize=16)
    # fig.text(0, 0.833, '$\mathbf{x} \sim p(\mathbf{x})$',
    #          rotation='vertical', verticalalignment='center', fontsize=16)
    # fig.text(0, 0.6, '$\mathbf{x} \sim \mathcal{CB}(\mathbf{x} | \mathbf{z}; \eta)$',
    #          rotation='vertical', verticalalignment='center', fontsize=16)
    # fig.text(0, 0.35, '$\mathbf{x} = \mathbb{E}_{\mathcal{B}}(\mathbf{x} | \mathbf{z}; \eta)$',
    #          rotation='vertical', verticalalignment='center', fontsize=16)
    # fig.text(0, 0.125, 'discrete$(\mathbf{x} = \eta)$',
    #          rotation='vertical', verticalalignment='center', fontsize=16)
    # fig.tight_layout()

    #######

    model_dist_b.eval()
    model_dist_b = model_dist_b.to(args_dist_b.device)

    train_loader = create_dataloader(train_data, args_dist_b)
    Z = []
    Y = []
    for batch_idx, (X, labels) in tqdm(enumerate(train_loader)):
        # Prepare
        X = X.to(args_dist_b.device)
        X = X.view([-1] + [X.shape[-2] * X.shape[-1]])

        mean, _ = model_dist_b.encoder(X)

        # Collect
        Z.append(mean)
        Y.append(labels)

    Z = torch.vstack(Z)
    Y = torch.hstack(Y)

    # Establish limits in the latent space
    mx = Z.max(0)[0].cpu().numpy()
    mn = Z.min(0)[0].cpu().numpy()

    fig = plot_latent_visualisation(model=model_dist_b, z_min=mn, z_max=mx, grid=(6, 6), img_dims=data_dims, Z_points=Z, labels=None, device=args_dist_b.device)

plt.show()

print("Done")