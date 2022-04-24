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
maze_dir = dataset_dir + 'only_grid_10000x11'

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

N = 42
Z = torch.randn(N, 2, device=args_dist_b.device)
bin_threshold = 0.5
binary_transform = BinaryTransform(bin_threshold)
with torch.inference_mode():
    # Sample the latents
    logits = model_dist_b.decoder(Z)

    # Use ancestral sampling to produce samples
    samples = model_dist_b.decoder.sample(logits, num_samples=1).detach().cpu()[0]
    means = model_dist_b.decoder.mean(logits).detach().cpu()
    # means = binary_transform(means)
    params = binary_transform(model_dist_b.decoder.param_p(logits).detach().cpu())

    # Create a matrix of samples
    X = torch.vstack([samples, means, params])
    X = X.view(-1, *data_dims).detach().cpu()
    X = X.squeeze()

    # Create a figure
    fig = plot_grid_of_samples(X, grid=(9, 14), figsize=(14, 10))
    fig.suptitle('VAE with Bernoulli Variational Distribution, trained on 10 000 11x11 Mazes.', fontsize=16)
    fig.text(0, 0.833, '$\mathbf{x} \sim \mathcal{CB}(\mathbf{x} | \mathbf{z}; \eta)$',
             rotation='vertical', verticalalignment='center', fontsize=16)
    fig.text(0, 0.5, '$\mathbf{x} = \mathbb{E}_{\mathcal{B}}(\mathbf{x} | \mathbf{z}; \eta)$',
             rotation='vertical', verticalalignment='center', fontsize=16)
    fig.text(0, 0.166, '$\mathbf{x} = \eta$ + discrete T',
             rotation='vertical', verticalalignment='center', fontsize=16)
    fig.tight_layout()

plt.show()