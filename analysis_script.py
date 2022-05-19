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
maze_dir = dataset_dir + 'only_grid_10000x27'#'only_grid_10000x11'
mnist_dir = dataset_dir #+ 'MNIST'
model_files = [
#    'VAE_cnn_maze10000x27_cnn_fc.pt',
#     'VAE_cnn_maze10000x27_fc_sd.pt',
#     'VAE_cnn_maze10000x27_fc_l2.pt',
#     'VAE_cnn_maze10000x27_l3_s.pt',
#     'VAE_cnn_maze10000x27_l3.pt',
#     'VAE_maze10000x27_6cnn_fc_bs50_e250.pt',
#     'VAE_maze10000x27_enc_6cnn_fc_dec_fc_2dconv_bs50_e250.pt',
#     'VAE_maze10000x27_6cnn_fc_bs50_e10.pt',
    'test.pt',
]

train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              #transforms.Resize((img_size, img_size)),
              transforms.ToTensor(),
          ]))

X = train_data[0][0]

img_size = 27

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

# Specify the hyperpameter choices

data_dims = (img_size,img_size,1)

for model_file in model_files:
    model_dist_b, optimiser_dist_b, args_dist_b = load_state('checkpoints/'+model_file, model_type=VAE, optim_type=optim.Adam)
    args_dist_b.cuda = False #CHANGE
    args_dist_b.device = torch.device("cuda" if args_dist_b.cuda else "cpu")
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

        train_loader = torch.utils.data.DataLoader(
            Maze_Dataset(
                maze_dir, train=True,
                transform=transforms.Compose([
                    #transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ])),
            batch_size=50,
            shuffle=True
        )

        Z = []
        Y = []
        STD = []

        for batch_idx, (X, labels) in tqdm(enumerate(train_loader)):
            # Prepare
            X = X.to(args_dist_b.device)
            X = X.squeeze() #TODO: figure out why X is sampled with shape [50,1,729,1]. train_data[0:10] is sampled with shape [1,10,729], but train_data.data has shape [10000, 729]

            mean, logvar = model_dist_b.encoder(X)

            # Collect
            Z.append(mean)
            Y.append(labels)
            STD.append(torch.exp(logvar/2))


        Z = torch.vstack(Z)
        Y = torch.hstack(Y)
        STD = torch.vstack(STD)

        # N = 1000
        # Z = 6*torch.randn(N, 2, device=args_dist_b.device)

        # Establish limits in the latent space
        mx = Z.max(0)[0].cpu().numpy()
        mn = Z.min(0)[0].cpu().numpy()

        print(f"Model: {model_file}")
        print(model_dist_b)
        print(args_dist_b)

        print(f"Mean of Encoder(X): Mean {Z.mean(axis=0)}, STD {Z.std(axis=0)}")
        print(f"Sigma of Encoder(X): Mean {STD.mean(axis=0)}, STD {STD.std(axis=0)}")

        print(f"Bounds on x axis: {(mn[0], mx[0])} / Bounds on y axis: {(mn[1], mx[1])}")

        figtitle = model_file + ' Latent space visualisation, Z ~ q(z|x)'
        fig = plot_latent_visualisation(model=model_dist_b, z_min=mn, z_max=mx, grid=(12, 12), img_dims=data_dims, Z_points=Z, labels=None, device=args_dist_b.device, alpha=0.2, title=figtitle)
        figtitle = model_file + ' Wide Latent space visualisation'
        fig2 = plot_latent_visualisation(model=model_dist_b, z_min=(-10,-10), z_max=(10,10), grid=(12, 12), img_dims=data_dims,
                                        Z_points=None, labels=None, device=args_dist_b.device, alpha=0.2, title=figtitle)

plt.show()

print("Done")