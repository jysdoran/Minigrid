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
maze_dir = dataset_dir + 'only_grid_10000x27'#'only_grid_10000x11'
mnist_dir = dataset_dir #+ 'MNIST'

img_size = 27

# #Uncomment
train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              #transforms.Resize((img_size, img_size)),
              transforms.ToTensor(),
          ]))

# train_data = torchvision.datasets.MNIST(
#     mnist_dir, train=True, download=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         BinaryTransform(0.6),
#         ]))
# test_data = torchvision.datasets.MNIST(
#     mnist_dir, train=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         BinaryTransform(0.6),
#         ]))


# plot some data
# indices = np.random.choice(len(train_data), size=64)
# samples = torch.vstack([train_data[idx][0] for idx in indices])
# #samples = samples.reshape(samples.shape[0], img_size, img_size, samples.shape[-1])
# samples = samples.reshape(samples.shape[0], img_size, img_size)
# fig = plot_grid_of_samples(samples, grid=(8,8))
# plt.show()

# VAE setup

parser = create_base_argparser()
parser = VAE.add_model_args(parser)
parser = CNNEncoder.add_extra_args(parser)
parser = dConvDecoder.add_extra_args(parser)
parser.print_help()

# Specify the hyperpameter choices
data_dim = train_data[0][0].numel()

# args_fc = ['--dec_layer_dims', '2', '130', f'{data_dim}',
#             '--dec_architecture', 'FC',
#              '--enc_architecture', 'FC',
#              '--enc_layer_dims', f'{data_dim}', '256', '32', '2',
#              '--gradient_type', 'pathwise',
#              '--num_variational_samples', '1',
#              '--data_distribution', 'Bernoulli',
#              '--epochs', '50',
#              '--learning_rate', '1e-4',
#              '--cuda']

# args_cnn_mnist = ['--dec_layer_dims', '2', '32', '64,13,13', '32,28,28', '1,28,28',
#             '--dec_architecture', 'dConv',
#             '--dec_kernel_size', '3',
#              '--enc_architecture', 'CNN',
#              '--enc_layer_dims', '1,28,28', '32,28,28', '64,14,14', '64,14,14', '64,14,14', '32', '2',
#              '--enc_kernel_size', '3',
#              '--gradient_type', 'pathwise',
#              '--num_variational_samples', '1',
#              '--data_distribution', 'Bernoulli',
#              '--epochs', '5',
#              '--learning_rate', '1e-4',
#              '--cuda']

# args_cnn = ['--dec_layer_dims', '2', '32', '32,13,13', '16,27,27', '1,27,27',
#             '--dec_architecture', 'dConv',
#             '--dec_kernel_size', '3',
#              '--enc_architecture', 'CNN',
#              '--enc_layer_dims', '1,27,27', '16,27,27', '32,14,14', '32', '2',
#              '--enc_kernel_size', '3',
#              '--gradient_type', 'pathwise',
#              '--num_variational_samples', '1',
#              '--data_distribution', 'Bernoulli',
#              '--epochs', '50',
#              '--learning_rate', '1e-4',
#              '--cuda']

#'--dec_layer_dims', '2', '16', '128', '128,6,6', '64,13,13', '64,27,27', '32,27,27', '1,27,27',
#'--dec_layer_dims', '2', '16', '128', '64,13,13', '64,27,27', '32,27,27', '1,27,27',
# '--dec_layer_dims', '2', '16', '128', '64,13,13', '32,27,27', '32,27,27', '32,27,27', '1,27,27',
# '--dec_layer_dims', '2', '16', '128', '64,13,13', '32,27,27', '32,27,27', '16,27,27', '1,27,27',
# '--dec_layer_dims', '2', '16', '128', '32,6,6', '8,13,13', '1,27,27', #no conv layer, need modification of decoder code

args_cnn_fc = ['--dec_layer_dims', '2', '16', '64,13,13', '32,27,27', '1,27,27',
            '--dec_architecture', 'dConv',
            '--dec_kernel_size', '3',
             '--enc_architecture', 'CNN',
             '--enc_layer_dims', '1,27,27', '32,27,27', '64,14,14', '64,14,14', '128,7,7', '128,7,7', '1024', '128', '16', '2',
             '--enc_kernel_size', '3',
             '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--epochs', '50',
             '--learning_rate', '1e-4',
             '--cuda']

save_as = 'VAE_maze10000x27_enc_6cnn_fc_dec_1fc_dconv_bs50_e50.pt'
args_dist_b = parser.parse_args(args_cnn_fc)# CHANGE
args_dist_b.batch_size = 50
#args_dist_b.seed = 10111201


args_dist_b.cuda = False#args_dist_b.cuda and torch.cuda.is_available() #change for enable
args_dist_b.device = torch.device("cuda" if args_dist_b.cuda else "cpu")

# Seed all random number generators for reproducibility of the runs
seed_everything(args_dist_b.seed)

# Initialise the model and the Adam (SGD) optimiser
model_dist_b = VAE(args_dist_b).to(args_dist_b.device)
optimizer_dist_b = optim.Adam(model_dist_b.parameters(), lr=args_dist_b.learning_rate)

model_dist_b, optimizer_dist_b, out_b, fig = fit_model(model_dist_b, optimizer_dist_b,
                                                      train_data, args_dist_b,
                                                      test_data=train_data)
save_file = 'checkpoints/' + save_as
save_state(args_dist_b, model_dist_b, optimizer_dist_b, save_file)
print(f"Saving to {save_file}")

print("Done")