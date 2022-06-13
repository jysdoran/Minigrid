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

CUDA = False

base_dir = str(Path(__file__).resolve().parent)
dataset_dir = base_dir + '/datasets/'
cifar_dir = dataset_dir + 'cifar10_data'
maze_dir = dataset_dir + 'multi_room10000x27'#'only_grid_10000x11'
mnist_dir = dataset_dir #+ 'MNIST'
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
    'test',
]

img_size = 27

train_data = Maze_Dataset(
          maze_dir, train=True,
          transform = transforms.Compose([
              SelectChannelsTransform(0),
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

data_dims = (1,img_size,img_size)

for model_name in model_names:
    tensorboard = SummaryWriter('runs/' + model_name)
    # address bug overwritting projector_config when a new embedding is created
    projector_config_file = Path.cwd() / tensorboard.log_dir / 'projector_config.pbtxt'
    projector_config_file.rename(str(projector_config_file)+'.bkup')

    if not CUDA: device = torch.device('cpu')
    else: device = None
    model_dist_b, optimiser_dist_b, args_dist_b = load_state('checkpoints/'+model_name+'.pt', model_type=VAE, optim_type=optim.Adam, device=device)
    args_dist_b.cuda = CUDA
    args_dist_b.device = torch.device("cuda" if args_dist_b.cuda else "cpu")
    bin_threshold = 0.5
    binary_transform = BinaryTransform(bin_threshold)
    flip_bits = FlipBinaryTransform()
    resize = torchvision.transforms.Resize((100, 100), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    train_loader = create_dataloader(train_data, args_dist_b)
    example_data_train, example_targets_train = next(iter(train_loader))  # TODO: make deterministic accross runs

    with torch.inference_mode():
        #######

        model_dist_b.eval()
        model_dist_b = model_dist_b.to(args_dist_b.device)

        # Compute latent space statistics
        # We don't use labels hence discard them with a _
        all_means = []
        all_stds = []
        for batch_idx, (mbatch, _) in enumerate(train_loader):
            mean, logvar = model_dist_b.encoder(mbatch)
            all_means.append(mean)
            all_stds.append(torch.exp(logvar/2))

        all_means = torch.cat(all_means)
        all_stds = torch.cat(all_stds)

        variational_mean = all_means.mean(axis=0)
        variational_std = all_stds.mean(axis=0)

        Z_interp, X_interp = latent_interpolation(model=model_dist_b, minibatch=example_data_train, Z_mean=variational_mean, Z_std=variational_std, dim_r_threshold=.2, n_interp=16, interpolation_scheme='polar', device=args_dist_b.device)
        X_interp = X_interp.reshape(X_interp.shape[0]*X_interp.shape[1], *X_interp.shape[2:])
        img_grid = torchvision.utils.make_grid(flip_bits(resize(X_interp)), nrow=Z_interp.shape[-2])
        #TODO store dimensionality of p(z|x) in TB
        tensorboard.add_image('Interpolated samples, Polar', img_grid)

        tensorboard.close()
        # address bug overwritting projector_config when a new embedding is created
        with open(str(projector_config_file) + '.temp', 'w') as fout, fileinput.input([str(projector_config_file) + '.bkup', str(projector_config_file)]) as fin:
            for line in fin:
                fout.write(line)
        # cleanup files
        new_projector_config_file = Path(str(projector_config_file) + '.temp').rename(str(projector_config_file))
        Path(str(projector_config_file)+'.bkup').unlink()
        sys.exit()

        # #Generate samples from prior - already done during training
        # Z = torch.randn(1024, *model_dist_b.decoder.bottleneck.input_size).to(args_dist_b.device) #M, B, D
        # logits = model_dist_b.decoder(Z)
        # generated_samples = model_dist_b.decoder.param_b(logits, bin_threshold)
        #
        # tensorboard.add_embedding(Z,
        #                           label_img=flip_bits(resize(generated_samples)),
        #                           tag='Generated_samples_from_prior', global_step=0)
        #
        # tensorboard.add_images('Generated_samples_from_prior', flip_bits(resize(generated_samples[0:64])), dataformats='NCHW')

        train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=False)
        # examples = iter(train_loader)
        # example_data, example_targets = examples.next()
        # metadata = example_targets.tolist()
        #
        # X = example_data.squeeze().to(args_dist_b.device)
        # mean, logvar = model_dist_b.encoder(X)
        #
        # writer.add_embedding(mean, metadata=metadata, label_img=example_data, tag='test_embedding', global_step=0)


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

        print(f"Model: {model_name}")
        print(model_dist_b)
        print(args_dist_b)

        print(f"Mean of Encoder(X): Mean {Z.mean(axis=0)}, STD {Z.std(axis=0)}")
        print(f"Sigma of Encoder(X): Mean {STD.mean(axis=0)}, STD {STD.std(axis=0)}")

        print(f"Bounds on x axis: {(mn[0], mx[0])} / Bounds on y axis: {(mn[1], mx[1])}")

        figtitle = model_name + ' Latent space visualisation, Z ~ q(z|x)'
        fig = plot_latent_visualisation(model=model_dist_b, z_min=mn, z_max=mx, grid=(12, 12), img_dims=data_dims, Z_points=Z, labels=None, device=args_dist_b.device, alpha=0.2, title=figtitle)
        figtitle = model_name + ' Wide Latent space visualisation'
        fig2 = plot_latent_visualisation(model=model_dist_b, z_min=(-10,-10), z_max=(10,10), grid=(12, 12), img_dims=data_dims,
                                        Z_points=None, labels=None, device=args_dist_b.device, alpha=0.2, title=figtitle)

plt.show()

print("Done")