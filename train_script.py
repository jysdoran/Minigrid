import sys

from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_loaders import Maze_Dataset
from util import *
from models.VAE import *

run_name = 'test'#'multiroom10000x27_6cnn_z12_b64e1000' #CHANGE
use_gpu = True
plot_every = 100
batch_size = 64
epochs = 2

writer = SummaryWriter('runs/' + run_name)
base_dir = str(Path(__file__).resolve().parent)
dataset_dir = base_dir + '/datasets/'
cifar_dir = dataset_dir + 'cifar10_data'
maze_dir = dataset_dir + 'multi_room10000x27'#'multi_room128x27'#'only_grid_128x27'
mnist_dir = dataset_dir #+ 'MNIST'

# #Uncomment
train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              SelectChannelsTransform(0),
              transforms.ToTensor(),
          ]))

test_data = Maze_Dataset(
          maze_dir, train=False,
          transform = transforms.Compose([
              SelectChannelsTransform(0),
              transforms.ToTensor(),
          ]))

# train_data = torchvision.datasets.MNIST(
#     mnist_dir, train=True, download=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         FlattenTransform(1,-1),
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

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

# VAE setup

parser = create_base_argparser()
parser = VAE.add_model_args(parser)
parser = CNNEncoder.add_extra_args(parser)
parser = dConvDecoder.add_extra_args(parser)
parser.print_help()

# Specify the hyperpameter choices
data_dim = train_data[0][0].numel()

args_fc = ['--dec_layer_dims', '2', '130', f'{data_dim}',
            '--dec_architecture', 'FC',
             '--enc_architecture', 'FC',
             '--enc_layer_dims', f'{data_dim}', '128', '32', '2',
             '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--epochs', str(epochs),
             '--learning_rate', '1e-4',
             '--cuda']

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
# try this: https://indico.cern.ch/event/996880/contributions/4188468/attachments/2193001/3706891/ChiakiYanagisawa_20210219_Conv2d_and_ConvTransposed2d.pdf

# graph_cnn_fc:
# input [10x10x4] (redundant entries) graph/layout obtained with binary(maxpool) or binary(pool) operation.
# enc:
# note: could experiment with stride 2 but not too recommended
# CNN1 k = 3, cin = 4, cout = 16 | weights : 576
# (max_pool) cout = 8
# CNN2 k = 3 cout = 64 | weights : 10k
# (max pool) cout = 32
# CNN3 k = 3 cout = 262144 (! 150M or 75M weights if we are linking all channels with each other. Do 1 to many channels here, then only 200k weights)
#            cout = 4096 | weights : 2.36M  (then can afford to do many to many channels)
# max pool cout = 256 | [weights updated/pass / 16!][divided by 16, works out nicely with option 2, as it seems to me there should be 16 possible symmetries (4mirror * 4 rotations)
# (CNN4 k = 4 cout = 4096) [! 16M weights + 4M weights for FC] - could consider skipping and going straight to flatten (4*4*256 -> 1024)
# FC 1024 -> 128 -> 8 (weights 4M, 100k, 1k)
# with (layers) removed, model works out to ~  6.5M however only <5M updated per pass. [not considering Relu]
# dec:
# FC 8 -> (128) -> (1024) -> 4096
# dconv1 k =4 cout = 256
# dconv2 k =3 cout = (32 if max pool in enc) / 64
# dconv3 k=3 cout = (8 if max pool in enc) / 16
# conv_same OR dconv [k = 3, cout = 4] OR maxpool [cout = 4] (consider cnn instead of dconv3 if maxpool)

# graph_conv1d [nxn] nodes
# input: [n][n-1][2] [n sequences of n-1 edges. 1 channel for h edge, 1 for v edges] (# using the minimum number of parameters for encoding)
# enc:
# CNN1 k = 3, cin = 2, cout = 8 | weights : 24
# CNN2 k = 3 cout = 64 | weights : 1536
# CNN3 k = 3 cout = 4096 | weights : 786k
# maxpool cout = 256
# flatten : [10x3x256]->[7680]
# FC 1024, 128, 8 | weights 7M, 100k, 1k
#total weights ~ 8M

# OR
# CNN3 k = 5, cout = 4096 [out=10*1*4096] | weights : 1.3M
# maxpool cout = 256
# flattent : [10*1*256] -> [2560]
# FC 1024, 128, 8 | weights 2.6M, 100k, 1k
# total weights ~ 4M

# dec:
# FC 8, 128, 7680
# reshape [10*3*256]
# dconv k = 3 cout=64 [10*5*64]

# OR
# FC 8, 128, 2560
# reshape [10*1*256]
# dconv k=5 cout = 64 [10*5*64]

# dconv k =3 cout=8 [10*7*8]
# dconv k = 3 cout = 8 + conv_same k = 3 OR #dconv k = 3 cout = 2 [10*9*2]

# max pool cout = 256 | [weights updated/pass / 16!][divided by 16, works out nicely with option 2, as it seems to me there should be 16 possible symmetries (4mirror * 4 rotations)
# (CNN4 k = 4 cout = 4096) [! 16M weights + 4M weights for FC] - could consider skipping and going straight to flatten (4*4*256 -> 1024)
# FC 1024 -> 128 -> 8 (weights 4M, 100k, 1k)
# with (layers) removed, model works out to ~  6.5M however only <5M updated per pass. [not considering Relu]
# dec:
# FC 8 -> (128) -> (1024) -> 4096
# dconv1 k =4 cout = 256
# dconv2 k =3 cout = (32 if max pool in enc) / 64
# dconv3 k=3 cout = (8 if max pool in enc) / 16
# conv_same OR dconv [k = 3, cout = 4] OR maxpool [cout = 4] (consider cnn instead of dconv3 if maxpool)


# graph_fc
# enc: 256 -> 128 -> 32 -> 8
# dec: 8 -> 32 -> 128 -> 256



#Enc: 288 + 18432 + 36864 + 36864 + 73728 + 73728 + 6.4M+ 1M + 1M + 131k + 131k+ 1k ~ 9M
args_cnn_fc = ['--dec_layer_dims', '12', '1024', '128,13,13', '64,27,27', '32,27,27', '1,27,27',
            '--dec_architecture', 'dConv',
            '--dec_kernel_size', '3',
             '--enc_architecture', 'CNN',
             '--enc_layer_dims', '1,27,27', '32,27,27', '64,14,14', '64,14,14', '128,7,7', '128,7,7', '1024', '1024', '128', '12',
             '--enc_kernel_size', '3',
             '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--epochs', str(epochs),
             '--learning_rate', '1e-4',
             '--cuda']

args_dist_b = parser.parse_args(args_cnn_fc)# CHANGE
args_dist_b.batch_size = batch_size
#args_dist_b.seed = 10111201

if use_gpu == True:
    args_dist_b.cuda = args_dist_b.cuda and torch.cuda.is_available()
else:
    args_dist_b.cuda = False
args_dist_b.device = torch.device("cuda" if args_dist_b.cuda else "cpu")

# Seed all random number generators for reproducibility of the runs
seed_everything(args_dist_b.seed)

# Initialise the model and the Adam (SGD) optimiser
model_dist_b = VAE(args_dist_b).to(args_dist_b.device)
optimizer_dist_b = optim.Adam(model_dist_b.parameters(), lr=args_dist_b.learning_rate)
# sys.exit()

model_dist_b, optimizer_dist_b, model_state_best_training_elbo, optim_state_best_training_elbo, early_t = fit_model(model_dist_b, optimizer_dist_b,
                                                      train_data, args_dist_b,
                                                      test_data=test_data, latent_eval_freq=plot_every, tensorboard=writer)
if early_t: run_name = run_name + '_early_t'
save_file = 'checkpoints/' + run_name + '.pt'
print(f"Saving to {save_file}")
save_state(args_dist_b, model_dist_b, optimizer_dist_b, save_file, [model_state_best_training_elbo], [optim_state_best_training_elbo])

writer.close()

print("Done")