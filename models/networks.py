from torch import nn, Tensor
from typing import Iterable, Tuple, Union
from models.layers import *
from math import prod
import numpy as np

import torch

class Network(nn.Module):
    """Fully connected PyTorch neural network class

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None):
        """Creates a network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        """
        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.layers = self.make_seq(dims, output_activation)

    def make_seq(self, dims: Iterable[int], output_activation: nn.Module) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass through the network

        :param x (torch.Tensor): input tensor to feed into the network
        :return (torch.Tensor): output computed by the network
        """
        # Feedforward
        return self.layers(x)


class FC_ReLU_Network(Network):

    """Fully connected PyTorch neural network class with ReLU hidden activations

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None):

        super().__init__(dims, output_activation)

    def make_seq(self, dims: Iterable[int], output_activation: nn.Module) -> nn.Module:

        """Creates a sequential network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)


class CNN_ReLU_Network(Network):

    """CNN PyTorch neural network class with ReLU hidden activations

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], output_activation: nn.Module = None):

        if not isinstance(kernel_sizes, int):
            assert len(kernel_sizes) == len(dims) - 1
        self.kernel_sizes = kernel_sizes
        self.channels = np.array(dims)[:, -1]
        self.img_dims = np.array(dims)[:, 0:2]
        self.strides = np.ceil(self.img_dims[0:-1]/self.img_dims[1:]).astype(int)

        super().__init__(dims, output_activation)

    def make_seq(self, dims: Iterable[Tuple[int, int, int]], output_activation: nn.Module) -> nn.Module:


        """Creates a sequential network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of channels
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 1):
            if isinstance(self.kernel_sizes, int):
                kernel_size = self.kernel_sizes
            else:
                kernel_size = self.kernel_sizes[i]
            mods.append(Conv2dSame(self.channels[i], self.channels[i+1], kernel_size, self.strides[i]))
            if i != (len(dims) - 2): #do not append ReLU layer on the last one.
                mods.append(nn.ReLU())

        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass through the network

        :param x (torch.Tensor): input tensor to feed into the network
        :return (torch.Tensor): output computed by the network
        """

        # Feedforward
        return self.layers(x)


class dConv_ReLU_Network(Network):

    """dConv PyTorch neural network class with ReLU hidden activations

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], output_activation: nn.Module = None):

        if not isinstance(kernel_sizes, int):
            assert len(kernel_sizes) == len(dims) - 1
        self.kernel_sizes = kernel_sizes
        self.channels = np.array(dims)[:, 0]
        self.img_dims = np.array(dims)[:, 1:]
        # check dimensions were correctly specified
        assert np.all(np.diff(self.img_dims, axis=0) % (self.kernel_sizes - 1) == 0)
        self.strides = (np.diff(self.img_dims,axis=0)/(self.kernel_sizes - 1)).astype(int)

        super().__init__(dims, output_activation)

    def make_seq(self, dims: Iterable[Tuple[int, int, int]], output_activation: nn.Module) -> nn.Module:

        """Creates a sequential network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of channels
            format : int, (CHANNEL, H, W)
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 1):
            if isinstance(self.kernel_sizes, int):
                kernel_size = self.kernel_sizes
            else:
                kernel_size = self.kernel_sizes[i]
            mods.append(nn.ConvTranspose2d(self.channels[i], self.channels[i+1], kernel_size, self.strides[i]))
            if i != (len(dims) - 2): #do not append ReLU layer on the last one.
                mods.append(nn.ReLU())

        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass through the network

        :param x (torch.Tensor): input tensor to feed into the network (N, C, H, W)
        :return (torch.Tensor): output computed by the network
        """

        # Feedforward
        x = self.layers(x)
        return x