import torch
import torchvision
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
        samples = samples.reshape(grid[0], grid[1], *samples.shape[1:])

    if len(samples.shape) == 5:
        samples = torch.permute(samples, (0, 1, 3, 4, 2)) #(Gx, Gy, C, H, W) -> (Gx, Gy, H, W, C)

    fig, axes = plt.subplots(grid[0], grid[1], sharex=True, sharey=True, figsize=figsize)
    fig.subplots_adjust(wspace=0., hspace=0.)

    samples = samples.cpu()

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
            axes[i, j].imshow(samples[i, j, ..., 0], cmap='gray_r')
            axes[i, j].imshow(samples[i, j, ..., 1], cmap=transparent_cmaps[0], vmin=0.9)
            axes[i, j].imshow(samples[i, j, ..., 2], cmap=transparent_cmaps[1], vmin=0.9)
        else:
            axes[i, j].imshow(samples[i, j], cmap='gray_r')
        axes[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    return fig, axes


# only for dim 2 at the moment
def plot_latent_visualisation(model, z_max: Tuple[float, float] = (10,10), z_min: Tuple[float, float] = (-10,-10), grid: Tuple[int, int] = (12,12),
                              img_dims: Tuple[int, int, int] = None, figsize: Tuple[int, int] = (10, 10), Z_points=None,
                              labels=None, device='cpu', alpha=0.5, title=None):
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
    if img_dims is not None:
        samples = samples.reshape(*samples.shape[:2], *img_dims)
    samples = samples.squeeze() #TODO: check if you can remove.

    fig, axes = plot_grid_of_samples(samples, grid=None, figsize=figsize)

    for i, j in itertools.product(range(grid[0]), range(grid[1])):
        if j == 0:
            axes[j, i].set_title(f'{x[i]:.2f}', fontsize=13)
        if i == 0:
            axes[j, i].set_ylabel(f'{y[grid[0] - j - 1]:.2f}', fontsize=13)

    if Z_points is not None:
        if isinstance(Z_points, torch.Tensor):
            Z_points = Z_points.detach().cpu().numpy()
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
        ax.scatter(Z_points[:, 0], Z_points[:, 1], color=c, alpha=alpha)
        ax.patch.set_alpha(0.)
        ax.set_xlim(z_min[0], z_max[0])
        ax.set_ylim(z_min[1], z_max[1])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    if title is not None:
        fig.suptitle(title, size=13)

    return fig

def latent_interpolation(model, minibatch:torch.Tensor, Z_mean=None, Z_std=None, n_interp:int = 4, dim_r_threshold=None, interpolation_scheme:str = 'linear', img_dims: Tuple[int, int, int] = None, figsize: Tuple[int, int] = (10, 10),
                              labels=None, device='cpu', alpha=0.5, title=None):

    assert len(minibatch) % 2 == 0

    if Z_mean is None: Z_mean = 0
    if Z_std is None: Z_std = 1

    model.eval()
    latents, _ = model.encoder(minibatch)

    # remove dimensions with average standard deviation of 1 (e.g. uninformative)
    if dim_r_threshold is not None:
        kept_dims = torch.where(Z_std < dim_r_threshold)[0].cpu().numpy()
        original_latents = latents.clone()
        latents = latents[..., kept_dims]
        Z_std_red = Z_std[kept_dims]
        Z_mean_red = Z_mean[kept_dims]
    else:
        Z_std_red = Z_std.clone()
        Z_mean_red = Z_mean.clone()

    latents = (latents - Z_mean_red) / Z_std_red


    if interpolation_scheme == 'linear':
        latents_dist = torch.cdist(latents, latents, p=2)
        eps = 5e-3
    if interpolation_scheme == 'polar':
        eps = 0.
        latents_dist = cdist_polar(latents, latents)

    latents_dist[latents_dist<=eps] = float('inf')
    dist, pair_indices = latents_dist.min(axis=0)
    dist = dist.cpu().numpy()
    pair_indices = pair_indices.cpu().numpy()

    distances = {}
    for ind_a, ind_b in enumerate(pair_indices):
        assert ind_a != ind_b
        key = (min(ind_a, ind_b), max(ind_a, ind_b))
        distances[key] = dist[ind_a]

    # sort the distances in ascending order
    sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    interpolation_points = np.linspace(0, 1, n_interp+2)[1:-1]
    interpolated_latents_dict = {}
    for pair in sorted_distances.keys():
        interpolated_latents = [latents[pair[0]],]
        for loc in interpolation_points:
            if interpolation_scheme == 'linear':
                interp = lerp(loc, latents[pair[0]], latents[pair[1]])
            elif interpolation_scheme == 'polar':
                interp = slerp(loc, latents[pair[0]], latents[pair[1]])
            interpolated_latents.append(interp)
        interpolated_latents.append(latents[pair[1]])
        interpolated_latents_dict[pair] = interpolated_latents

    # convert back to tensor
    l_t = [torch.stack(i) for i in list(interpolated_latents_dict.values())]
    interpolated_latents_grid = torch.stack(l_t).to(device)

    # Go back to original sample
    interpolated_latents_grid = (interpolated_latents_grid * Z_std_red) + Z_mean_red

    #reassembly
    if dim_r_threshold is not None:
        interpolated_latents_grid_r = torch.zeros(*interpolated_latents_grid.shape[:-1], original_latents.shape[-1], dtype=torch.float).to(device)
        interpolated_latents_grid_r[..., kept_dims] = interpolated_latents_grid
    else:
        interpolated_latents_grid_r = interpolated_latents_grid

    # Obtain the interpolated samples
    interpolated_logits = model.decoder(interpolated_latents_grid_r)
    interpolated_samples = model.decoder.param_b(interpolated_logits)


    #TODO decode back before plotting
    # use the plot samples function

    return interpolated_latents_grid, interpolated_samples




    logits = model.decoder(torch.tensor(Z_grid, dtype=torch.float, device=device))
    samples = model.decoder.param_b(logits).detach().cpu()
    if img_dims is not None:
        samples = samples.reshape(*samples.shape[:2], *img_dims)
    samples = samples.squeeze() #TODO: check if you can remove.

    fig, axes = plot_grid_of_samples(samples, grid=None, figsize=figsize)

    for i, j in itertools.product(range(grid[0]), range(grid[1])):
        if j == 0:
            axes[j, i].set_title(f'{x[i]:.2f}', fontsize=13)
        if i == 0:
            axes[j, i].set_ylabel(f'{y[grid[0] - j - 1]:.2f}', fontsize=13)

    if Z_points is not None:
        if isinstance(Z_points, torch.Tensor):
            Z_points = Z_points.detach().cpu().numpy()
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
        ax.scatter(Z_points[:, 0], Z_points[:, 1], color=c, alpha=alpha)
        ax.patch.set_alpha(0.)
        ax.set_xlim(z_min[0], z_max[0])
        ax.set_ylim(z_min[1], z_max[1])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    if title is not None:
        fig.suptitle(title, size=13)

    return fig

def lerp(val, low, high):
    return (1.0-val) * low + val * high

def slerp(val, low, high):
    omega = torch.arccos(torch.clip(torch.dot(low/torch.linalg.norm(low), high/torch.linalg.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return lerp(val, low, high) # L'Hopital's rule/LERP
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high

def slerp2(val, low, high):
    low_n = np.linalg.norm(low)
    high_n = np.linalg.norm(high)
    omega = np.arccos(np.clip(np.dot(low/low_n, high/high_n), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return lerp(val, low, high) # L'Hopital's rule/LERP
    return (np.sin((1.0-val)*omega) / so * low / low_n + np.sin(val*omega) / so * high / high_n) * ((1.0 - val)*low_n + val * high_n)

# "ellipse lerp": will work for vectors that do not have the same norm
def elerp(val, low, high):
    low_n = np.linalg.norm(low)
    high_n = np.linalg.norm(high)
    omega = np.arccos(np.clip(np.dot(low/low_n, high/high_n), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return lerp(val, low, high) # L'Hopital's rule/LERP
    return slerp(val, low / low_n, high / high_n) * lerp(val, low_n, high_n)

# only gives sensible values if low and high both have unit norm
def nlerp(val, low, high):
    p = lerp(val, low, high)
    return p / np.linalg.norm(p)

# broken
def celerp(val, low, high, centroid):
    #centroid = (low + high)/2
    low = low - centroid
    high = high - centroid
    return elerp(val, low, high) + centroid

def cdist_polar(x1, x2, eps=1e-08):
    return torch.abs(torch.nn.functional.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1, eps=eps))



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
    parser.add_argument('--data_dims', type=tuple, default=(1,27,27),
                        help='Input and output data dimensions')
    return parser


class BinaryTransform(object):
    def __init__(self, thr):
        self.thr = thr

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


class FlipBinaryTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.abs(x - 1)


class ReshapeTransform(object):
    def __init__(self, *args):
        self.shape = args

    def __call__(self, x: torch.Tensor):
        return x.view(*self.shape)


class FlattenTransform(object):
    def __init__(self, *args):
        self.dims = args # start and end dim

    def __call__(self, x: torch.Tensor):
        return x.view(*self.dims)


class SelectChannelsTransform(object):
    def __init__(self, *args):
        self.selected_channels = args

    def __call__(self, x: torch.Tensor):
        return x[..., self.selected_channels]


class ToDeviceTransform(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, x: torch.Tensor):
        return x.to(self.device)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            data, targets = b
            yield (self.func(data), self.func(targets))

    @property
    def dataset(self):
        return self.dl.dataset


### Kind of VAE specific:

def layer_dim(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        try:
            return (int(s),)
        except:
            raise argparse.ArgumentTypeError("Each tuple must be x,y,z")

def per_datapoint_elbo_to_avgelbo_and_loss(elbos):
    # Compute the average ELBO over the mini-batch
    elbo = elbos.mean(0)
    # We want to _maximise_ the ELBO, but the SGD implementations
    # do minimisation by default, hence we multiply the ELBO by -1.
    loss = -elbo

    return elbo, loss


def create_dataloader(data, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, **kwargs)

    wrapped_data_loader = WrappedDataLoader(data_loader, ToDeviceTransform(args.device))

    return wrapped_data_loader


def fit_model(model, optimizer, train_data, args, *, test_data=None, tensorboard=None, latent_eval_freq=10):
    flip_bits = FlipBinaryTransform()
    resize = torchvision.transforms.Resize((100, 100), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    # Create data loaders
    train_loader = create_dataloader(train_data, args)
    example_data_train, example_targets_train = next(iter(train_loader)) #TODO: make deterministic accross runs
    target_metadata_train = example_targets_train.tolist() #TODO: change depending on dataset
    if tensorboard is not None:
        tensorboard.add_text('Encoder Architecture', str(model.encoder.model))
        tensorboard.add_text('Bottleneck Architecture', str(model.encoder.mean))
        tensorboard.add_text('Decoder Architecture', str(model.decoder.model))
        tensorboard.add_text('Hyperparameters', str(args))
        tensorboard.add_images('train_samples', flip_bits(resize(example_data_train)), dataformats='NCHW')
        tensorboard.add_graph(model, example_data_train)
    if test_data is not None:
        test_loader = create_dataloader(test_data, args)
        example_data_test, example_targets_test = next(iter(test_loader))
        target_metadata_test = example_targets_test.tolist()  # TODO: change depending on dataset
        if tensorboard is not None:
            img_grid = torchvision.utils.make_grid(flip_bits(example_data_test))
            tensorboard.add_images('train_samples', flip_bits(resize(example_data_test)), dataformats='NCHW')

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
            #mbatch = mbatch.to(args.device)
            # Flatten the images
            #mbatch = mbatch.view([-1] + [mbatch.shape[-2] * mbatch.shape[-1]])
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
                    #mbatch = mbatch.to(args.device)
                    # Flatten the images
                    #mbatch = mbatch.view([-1] + [mbatch.shape[-2] * mbatch.shape[-1]])

                    # Compute the loss for the test mini-batch
                    elbo, loss = per_datapoint_elbo_to_avgelbo_and_loss(model(mbatch))

                    epoch_test_elbos += [elbo.detach().item()]
                    pbar.set_description((f'Test Epoch: {epoch} [{batch_idx * len(mbatch)}/{len(test_loader.dataset)} '
                                          f'({100. * batch_idx / len(test_loader):.0f}%)] ELBO: {elbo:.6f}'))

        # Store epoch summary in list
        epoch_avg_train_elbo = np.mean(epoch_train_elbos)
        train_avg_epochs += [epoch]
        train_avg_elbos += [epoch_avg_train_elbo]
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

        # Tensorboard tracking
        if tensorboard is not None:
            example_data = example_data_train
            example_targets = example_targets_train
            target_metadata = target_metadata_train
            loader = train_loader
            epoch_avg_elbo = epoch_avg_train_elbo

            tensorboard.add_scalar('train ELBO', epoch_avg_elbo, epoch * len(loader))

            mean, logvar = model.encoder(example_data)
            std_of_abs_mean = torch.linalg.norm(mean, dim=1).std().item()
            mean_of_abs_std = logvar.exp().sum(axis=1).sqrt().mean().item()
            tensorboard.add_scalars('Train data Encoder stats vs steps', {'std(||mean(z)||), z~q(z|x)': std_of_abs_mean,
                                                                  'E[std(z)], z~q(z|x)': mean_of_abs_std,}, epoch * len(loader))

            if epoch % latent_eval_freq == 0:
                # latent_space_vis = plot_latent_visualisation(model, Z_points=mean, labels=example_targets, device=args.device)
                # tensorboard.add_figure('Latent space visualisation, Z ~ q(z|x), x ~ train data', latent_space_vis, global_step=epoch)

                tensorboard.add_embedding(mean, metadata=target_metadata, label_img=flip_bits(resize(example_data)),
                                          tag='train_data_encoded_samples', global_step=epoch)

                logits = model.decoder(mean)
                decoded_samples = model.decoder.param_b(logits)
                tensorboard.add_embedding(mean, metadata=target_metadata, label_img=flip_bits(resize(decoded_samples)),
                                          tag='train_data_decoded_samples', global_step=epoch)

            if test_data is not None:
                example_data = example_data_test
                example_targets = example_targets_test
                target_metadata = target_metadata_test
                epoch_avg_elbo = epoch_avg_test_elbo
                tensorboard.add_scalar('test ELBO', epoch_avg_elbo, epoch * len(loader))

                mean, logvar = model.encoder(example_data)
                std_of_abs_mean = torch.linalg.norm(mean, dim=1).std().item()
                mean_of_abs_std = logvar.exp().sum(axis=1).sqrt().mean().item()
                tensorboard.add_scalars('Test data Encoder stats vs steps', {'std(||mean(z)||), z~q(z|x)': std_of_abs_mean,
                                                                     'E[std(z)], z~q(z|x)': mean_of_abs_std, }, epoch * len(loader))

                if epoch % latent_eval_freq == 0:
                    # latent_space_vis = plot_latent_visualisation(model, Z_points=mean, labels=example_targets,
                    #                                              device=args.device)
                    # tensorboard.add_figure('Latent space visualisation, Z ~ q(z|x), x ~ test data', latent_space_vis,
                    #                        global_step=epoch)

                    tensorboard.add_embedding(mean, metadata=target_metadata,
                                              label_img=flip_bits(resize(example_data)),
                                              tag='test_data_encoded_samples', global_step=epoch)

                    logits = model.decoder(mean)
                    decoded_samples = model.decoder.param_b(logits)
                    tensorboard.add_embedding(mean, metadata=target_metadata,
                                              label_img=flip_bits(resize(decoded_samples)),
                                              tag='test_data_decoded_samples', global_step=epoch)

    # Reset gradient computations in the computation graph
    optimizer.zero_grad()

    if best_model_state is not None and best_epoch != args.epochs:
        print(f'Loading best model state from epoch {best_epoch}.')
        model.load_state_dict(best_model_state)
    if best_optim_state is not None and best_epoch != args.epochs:
        print(f'Loading best optimizer state from epoch {best_epoch}.')
        optimizer.load_state_dict(best_optim_state)

    if tensorboard is not None:
        Z = torch.randn(1024, *model.decoder.bottleneck.input_size).to(args.device) #M, B, D
        logits = model.decoder(Z)
        generated_samples = model.decoder.param_b(logits)
        tensorboard.add_embedding(Z,
                                  label_img=flip_bits(resize(generated_samples)),
                                  tag='Generated_samples_from_prior', global_step=0)
        tensorboard.add_images('Generated_samples_from_prior', flip_bits(resize(generated_samples[0:64])), dataformats='NCHW')

        example_data = example_data_train
        example_targets = example_targets_train
        target_metadata = target_metadata_train

        mean, logvar = model.encoder(example_data_train)
        logits = model.decoder(mean)
        decoded_samples = model.decoder.param_b(logits)
        tensorboard.add_embedding(mean, metadata=target_metadata, label_img=flip_bits(resize(example_data)),
                                  tag='train_data_encoded_samples', global_step=0)
        tensorboard.add_embedding(mean, metadata=target_metadata, label_img=flip_bits(resize(decoded_samples)),
                                  tag='train_data_decoded_samples', global_step=0)

        tensorboard.add_images('decoded_train_samples', flip_bits(resize(decoded_samples)), dataformats='NCHW')

        if test_data is not None:
            example_data = example_data_test
            example_targets = example_targets_test
            target_metadata = target_metadata_test
            epoch_avg_elbo = epoch_avg_test_elbo

            mean, logvar = model.encoder(example_data)
            logits = model.decoder(mean)
            decoded_samples = model.decoder.param_b(logits)

            tensorboard.add_embedding(mean, metadata=target_metadata,
                                      label_img=flip_bits(resize(example_data)),
                                      tag='test_data_encoded_samples', global_step=0)
            tensorboard.add_embedding(mean, metadata=target_metadata,
                                      label_img=flip_bits(resize(decoded_samples)),
                                      tag='test_data_decoded_samples', global_step=0)
            tensorboard.add_images('decoded_test_samples', flip_bits(resize(decoded_samples)), dataformats='NCHW')

    return model, optimizer


def save_state(args, model, optimizer, file):
    return torch.save({
        'argparser': args,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file)


def load_state(file, model=None, optimizer=None, model_type=None, optim_type=None, device=None):
    checkpoint = torch.load(file, map_location=device)
    parser = checkpoint['argparser']
    if device is not None: parser.device = device
    if model is None:
        model = model_type(parser).to(parser.device)
        optimizer = optim_type(model.parameters(), lr=parser.learning_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, parser