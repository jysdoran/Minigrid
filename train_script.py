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

# data_loader = torch.utils.data.DataLoader(
#     Maze_Dataset(
#           maze_dir, train=True,
#           transform = transforms.Compose([
#               transforms.Resize((img_size, img_size)),
#               transforms.ToTensor(),
#           ])),
#           batch_size=2,
#           shuffle=True
#           )


train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              #transforms.Resize((img_size, img_size)),
              transforms.ToTensor(),
          ]))

#indices = np.random.choice(len(train_data), size=64)
#samples = torch.vstack([train_data[idx][0] for idx in indices])
#samples = samples.reshape(samples.shape[0], img_size, img_size, samples.shape[-1])
#fig = plot_grid_of_samples(samples, grid=(8,8))
#plt.show()

# VAE setup

parser = create_base_argparser()
parser = VAE.add_model_args(parser)
parser.print_help()

# Specify the hyperpameter choices
data_dim = train_data[0][0].numel()
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

model_dist_b, optimizer_dist_b, out_b, fig = fit_model(model_dist_b, optimizer_dist_b,
                                                      train_data, args_dist_b,
                                                      test_data=train_data)

save_state(model_dist_b, optimizer_dist_b, 'checkpoints/VAE_discrete_onlygrid_10.pt')

print("Done")