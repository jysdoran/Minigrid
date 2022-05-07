import torch
import os
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import itertools
from tqdm import tqdm
from copy import deepcopy
from typing import Tuple


def seed_everything(seed=20211201):
    """
    Helper to seed random number generators.

    Note, however, that even with the seeds fixed, some non-determinism is possible.

    For more details read <https://pytorch.org/docs/stable/notes/randomness.html>.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # The below might help to make it more deterministic
    # but will hurt the performance significantly
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def plot_grid_of_samples(samples, grid=None, figsize=(8, 8)):
    """A helper function for plotting samples"""
    if grid is None:
        grid = (samples.shape[0], samples.shape[1])
    else:
        samples = samples.reshape(grid[0], grid[1], *samples.shape[2:])

    fig, axes = plt.subplots(grid[0], grid[1], sharex=True, sharey=True, figsize=figsize)
    fig.subplots_adjust(wspace=0., hspace=0.)

    if samples.shape[-1] == 3:
        cmaps = [plt.get_cmap('Reds'), plt.get_cmap('Greens')]
        transparent_cmaps = []
        for cmap in cmaps:
            color_array = cmap(range(256))
            color_array[:, -1] = np.linspace(0.0, 1.0, 256)
            new_cmap_name = str(cmap.name) + '_alpha'
            new_cmap = LinearSegmentedColormap.from_list(name=new_cmap_name, colors=color_array)
            transparent_cmaps.append(new_cmap)

    # Plot samples
    for i, j in itertools.product(range(grid[0]), range(grid[1])):
        if samples.shape[-1] == 3:
            axes[j, i].imshow(samples[i, grid[0] - j - 1, ..., 0], cmap='gray_r')
            axes[j, i].imshow(samples[i, grid[0] - j - 1, ..., 1], cmap=transparent_cmaps[0], vmin=0.9)
            axes[j, i].imshow(samples[i, grid[0] - j - 1, ..., 2], cmap=transparent_cmaps[1], vmin=0.9)
        else:
            axes[j, i].imshow(samples[i, grid[0] - j - 1], cmap='gray_r')
        axes[j, i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    return fig, axes


# only for dim 2 at the moment
def plot_latent_visualisation(model, z_max: Tuple[float, float], z_min: Tuple[float, float], grid: Tuple[int, int],
                              img_dims: Tuple[int, int, int], figsize: Tuple[int, int] = (14, 14), Z_points=None,
                              labels=None, device='cpu'):
    #TODO: finish implementation with labels
    #TODO: profiling to figure out why so slow for larger grids

    # Locations of the grid samples
    x = np.linspace(z_min[0], z_max[0], grid[0])
    y = np.linspace(z_min[1], z_max[1], grid[1])

    # Create the tuples of latent locations
    x0, x1 = np.meshgrid(x,y)
    Z_grid = np.empty(x0.shape + (2,))
    Z_grid[:, :, 0] = x0
    Z_grid[:, :, 1] = x1

    model.eval()
    logits = model.decoder(torch.tensor(Z_grid, dtype=torch.float, device=device))
    samples = model.decoder.param_b(logits).detach().cpu()
    samples = samples.reshape(*samples.shape[:2], *img_dims).squeeze()

    fig, axes = plot_grid_of_samples(samples, grid=None, figsize=figsize)

    for i, j in itertools.product(range(grid[0]), range(grid[1])):
        if j == 0:
            axes[j, i].set_title(f'{x[i]:.2f}', fontsize=13)
        if i == 0:
            axes[j, i].set_ylabel(f'{y[grid[0] - j - 1]:.2f}', fontsize=13)

    if Z_points is not None:
        if isinstance(Z_points, torch.Tensor):
            Z_points = Z_points.cpu().numpy()
        # Overlay another axes
        rect = [axes[0][0].get_position().get_points()[0, 0], axes[-1][-1].get_position().get_points()[0, 1],
                axes[-1][-1].get_position().get_points()[1, 0] - axes[0][0].get_position().get_points()[0, 0],
                axes[0][0].get_position().get_points()[1, 1] - axes[-1][-1].get_position().get_points()[0, 1]
                ]
        ax = fig.add_axes(rect)

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            # Create legend
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            c = np.array(colors)[0]  # TODO: Y.numpy()
            legend_elements = [mpl.patches.Patch(facecolor=colors[i], label=i) for i in range(10)]
            ax.legend(handles=legend_elements, ncol=10,
                      bbox_to_anchor=(0.5, 0.92),
                      bbox_transform=fig.transFigure,
                      loc='center',
                      prop={'size': 14},
                      frameon=False)
        else:
            c = 'g'
        # Plot projections
        ax.scatter(Z_points[:, 0], Z_points[:, 1], color=c, alpha=0.5)
        ax.patch.set_alpha(0.)
        ax.set_xlim(z_min[0], z_max[0])
        ax.set_ylim(z_min[1], z_max[1])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    return fig


def create_base_argparser():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--seed', type=int, default=20211201,
                        help='Random seed for reproducible runs.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Uses CUDA training if available (default: False)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs to train (default: 3000)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the Adam optimiser (default: 0.0001)')
    return parser


class BinaryTransform(object):
    def __init__(self, thr):
        self.thr = thr  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


### Kind of VAE specific:

def per_datapoint_elbo_to_avgelbo_and_loss(elbos):
    # Compute the average ELBO over the mini-batch
    elbo = elbos.mean(0)
    # We want to _maximise_ the ELBO, but the SGD implementations
    # do minimisation by default, hence we multiply the ELBO by -1.
    loss = -elbo

    return elbo, loss


def create_dataloader(data, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, **kwargs)


def fit_model(model, optimizer, train_data, args, *, test_data=None):
    # We will plot the learning curves during training
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Create data loaders
    train_loader = create_dataloader(train_data, args)
    if test_data is not None:
        test_loader = create_dataloader(test_data, args)

    train_epochs = []
    train_elbos = []
    train_avg_epochs = []
    train_avg_elbos = []
    test_avg_epochs = []
    test_avg_elbos = []

    # We will use these to track the best performing model on test data
    best_avg_test_elbo = float('-inf')
    best_epoch = None
    best_model_state = None
    best_optim_state = None

    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        # Train
        model.train()
        epoch_train_elbos = []
        # We don't use labels hence discard them with a _
        for batch_idx, (mbatch, _) in enumerate(train_loader):
            mbatch = mbatch.to(args.device)
            # Flatten the images
            mbatch = mbatch.view([-1] + [mbatch.shape[-2] * mbatch.shape[-1]])
            # Reset gradient computations in the computation graph
            optimizer.zero_grad()

            # Compute the loss for the mini-batch
            elbo, loss = per_datapoint_elbo_to_avgelbo_and_loss(model(mbatch))

            # Compute the gradients using backpropagation
            loss.backward()
            # Perform an SGD update
            optimizer.step()

            epoch_train_elbos += [elbo.detach().item()]
            pbar.set_description((f'Train Epoch: {epoch} [{batch_idx * len(mbatch)}/{len(train_loader.dataset)}'
                                  f'({100. * batch_idx / len(train_loader):.0f}%)] ELBO: {elbo:.6f}'))

        # Test
        if test_data is not None:
            with torch.inference_mode():
                model.eval()
                epoch_test_elbos = []
                for batch_idx, (mbatch, _) in enumerate(test_loader):
                    mbatch = mbatch.to(args.device)
                    # Flatten the images
                    mbatch = mbatch.view([-1] + [mbatch.shape[-2] * mbatch.shape[-1]])

                    # Compute the loss for the test mini-batch
                    elbo, loss = per_datapoint_elbo_to_avgelbo_and_loss(model(mbatch))

                    epoch_test_elbos += [elbo.detach().item()]
                    pbar.set_description((f'Test Epoch: {epoch} [{batch_idx * len(mbatch)}/{len(test_loader.dataset)} '
                                          f'({100. * batch_idx / len(test_loader):.0f}%)] ELBO: {elbo:.6f}'))

        # Store epoch summary in list
        train_avg_epochs += [epoch]
        train_avg_elbos += [np.mean(epoch_train_elbos)]
        train_epochs += np.linspace(epoch - 1, epoch, len(epoch_train_elbos)).tolist()
        train_elbos += epoch_train_elbos
        if test_data is not None:
            test_avg_epochs += [epoch]
            epoch_avg_test_elbo = np.mean(epoch_test_elbos)
            test_avg_elbos += [epoch_avg_test_elbo]

            # Snapshot best model
            if epoch_avg_test_elbo > best_avg_test_elbo:
                best_avg_test_elbo = epoch_avg_test_elbo
                best_epoch = epoch

                best_model_state = deepcopy(model.state_dict())
                best_optim_state = deepcopy(optimizer.state_dict())

        # Update learning curve figure
        ax.clear()
        ax.plot(train_epochs, train_elbos, color='b', alpha=0.5, label='train')
        ax.plot(np.array(train_avg_epochs) - 0.5, train_avg_elbos, color='b', label='train (avg)')
        if len(test_avg_elbos) > 0:
            ax.plot(np.array(test_avg_epochs) - 0.5, test_avg_elbos, color='r', label='test (avg)')
        ax.grid(True)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.legend(loc='lower right')
        ax.set_ylabel('ELBO')
        ax.set_xlabel('Epoch')

        fig_title = f'Epoch: {epoch}, Avg. train ELBO: {np.mean(epoch_train_elbos):.2f}, Avg. test ELBO: {np.mean(epoch_test_elbos):.2f}'
        # If we are tracking best model, then also highlight it on the plot and figure title
        if best_avg_test_elbo != float('-inf'):
            fig_title += f', Best avg. test ELBO: {best_avg_test_elbo:.2f}'
            ax.scatter(best_epoch - 0.5, best_avg_test_elbo, marker='*', color='r')

        fig.suptitle(fig_title, size=13)
        fig.tight_layout()
        # display.clear_output(wait=True) #TODO: check
        # if epoch != args.epochs:
        #     # Force display of the figure (except last epoch, where
        #     # jupyter automatically shows the contained figure)
        #     display.display(fig)

    # Reset gradient computations in the computation graph
    optimizer.zero_grad()

    if best_model_state is not None and best_epoch != args.epochs:
        print(f'Loading best model state from epoch {best_epoch}.')
        model.load_state_dict(best_model_state)
    if best_optim_state is not None and best_epoch != args.epochs:
        print(f'Loading best optimizer state from epoch {best_epoch}.')
        optimizer.load_state_dict(best_optim_state)

    out = {
        'train_avg_epochs': train_avg_epochs,
        'train_avg_elbos': train_avg_elbos,
        'train_epochs': train_epochs,
        'train_elbos': train_elbos,
        'test_avg_epochs': test_avg_epochs,
        'test_avg_elbos': test_avg_elbos
    }
    return model, optimizer, out, fig


def save_state(model, optimizer, file):
    return torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file)


def load_state(model, optimizer, file):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return
