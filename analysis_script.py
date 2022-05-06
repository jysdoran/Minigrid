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
#args_dist_b.cuda = args_dist_b.cuda and torch.cuda.is_available()
args_dist_b.cuda = False
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

    #############

    # logits = model_dist_b.decoder(Z)
    # samples = binary_transform(model_dist_b.decoder.param_p(logits).detach().cpu())
    # samples = samples.view(-1, *data_dims)
    # samples = samples.squeeze()
    #
    # Z_grid = torch.randn(20, 20, 2, device=args_dist_b.device)
    #
    # logits2 = model_dist_b.decoder(torch.tensor(Z_grid, dtype=torch.float, device=args_dist_b.device)).cpu()
    # logits2 = model_dist_b.decoder.param_p(logits2).detach().cpu()
    # samples2 = binary_transform(logits2).reshape(Z_grid.shape[0], Z_grid.shape[1], *data_dims)
    # samples2 = samples2.reshape((Z_grid.shape[0]**2, samples2.shape[2], samples2.shape[3]))
    #
    # fig = plot_grid_of_samples(samples, grid=(3,14), figsize=(14,10))
    # fig.tight_layout()
    # fig = plot_grid_of_samples(samples2[:N], grid=(3, 14), figsize=(14, 10))
    # fig.tight_layout()

    #plt.show()

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


    # #######
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # fig.suptitle('Discrete MNIST with Bernoulli Distribution', fontsize=16)
    #
    # # Transfer tensors back to cpu
    # Z = Z.cpu()
    # Y = Y.cpu()
    #
    # # Plot all samples in a specific color dependent on the label
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # c = np.array(colors)[0] #TODO: Y.numpy()
    # ax.scatter(Z[:, 0], Z[:, 1], color=c, alpha=0.3)
    #
    # # Create legend
    # legend_elements = [mpl.patches.Patch(facecolor=colors[i], label=i) for i in range(10)]
    # ax.legend(handles=legend_elements, ncol=5,
    #           bbox_to_anchor=(0.5, 0.93),
    #           bbox_transform=fig.transFigure,
    #           loc='center',
    #           prop={'size': 14},
    #           frameon=False)
    #
    # fig.tight_layout()

######

# Establish limits in the latent space
mx = Z.max(0)[0].cpu().numpy()
mn = Z.min(0)[0].cpu().numpy()

# Define the number of steps along each dimension
steps = 5
step_x = (mx[0] - mn[0]) / steps
step_y = (mx[1] - mn[1]) / steps

# Locations of the grid samples
x = np.arange(mn[0], mx[0], step_x)
y = np.arange(mn[1], mx[1], step_y)

# Create the tuples of latent locations
x0, x1 = np.mgrid[mn[0]:mx[0]:step_x,
         mn[1]:mx[1]:step_y]
Z_grid = np.empty(x0.shape + (2,))
Z_grid[:, :, 0] = x0
Z_grid[:, :, 1] = x1

with torch.inference_mode():
    # Generate samples
    logits = model_dist_b.decoder(torch.tensor(Z_grid, dtype=torch.float, device=args_dist_b.device)).cpu()
    samples = binary_transform(logits).reshape(steps, steps, *data_dims).squeeze()

    fig, axes = plot_grid_of_samples(samples, grid=None, figsize=(14, 14))

    grid = (steps, steps)

    for i, j in itertools.product(range(grid[0]), range(grid[1])):
        if j == 0:
            axes[j, i].set_title(f'{x[i]:.2f}', fontsize=13)
        if i == 0:
            axes[j, i].set_ylabel(f'{y[grid[0] - j - 1]:.2f}', fontsize=13)

    # Overlay another axes
    rect = [axes[0][0].get_position().get_points()[0, 0], axes[-1][-1].get_position().get_points()[0, 1],
            axes[-1][-1].get_position().get_points()[1, 0] - axes[0][0].get_position().get_points()[0, 0],
            axes[0][0].get_position().get_points()[1, 1] - axes[-1][-1].get_position().get_points()[0, 1]
            ]
    ax = fig.add_axes(rect)

    # Plot projections
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c = np.array(colors)[0] #TODO: Y.numpy()
    ax.scatter(Z[:, 0], Z[:, 1], color=c, alpha=0.25)
    ax.patch.set_alpha(0.)
    ax.set_xlim(mn[0], mn[0] + step_x * steps)
    ax.set_ylim(mn[1], mn[1] + step_y * steps)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Create legend
    legend_elements = [mpl.patches.Patch(facecolor=colors[i], label=i) for i in range(10)]
    ax.legend(handles=legend_elements, ncol=10,
              bbox_to_anchor=(0.5, 0.92),
              bbox_transform=fig.transFigure,
              loc='center',
              prop={'size': 14},
              frameon=False)

plt.show()

print("Done")

# questions:
# 1. check why output is not in (0,1)
# 2. check why different outcomes when plotting using plot_samples or from the Z_grid method
# 3. also patterns look kind of different when they should be the same.f