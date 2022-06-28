from torch import nn, Tensor
from typing import Iterable, Tuple, Union
from models.layers import *
from math import prod
import numpy as np

import torch

class Network(nn.Module):
    """PyTorch neural network class

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
            mods.append(nn.Linear(prod(dims[i]), prod(dims[i + 1])))
            mods.append(nn.ReLU())

        mods.append(nn.Linear(prod(dims[-2]), prod(dims[-1])))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

#class CNN_Factory:
class CNN_Factory(Network):

    """Factory for PyTorch neural network classes, will instanciate a new CNN instance based on input arguments

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __new__(cls, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], strides: Union[Iterable[int], int],
            arch: str = 'CNN', output_activation: nn.Module = None, same_padding: bool = False):

        if arch == 'CNN':
            if same_padding:
                instance = super().__new__(Same_CNN_ReLU_Network)
            else:
                instance = super().__new__(CNN_ReLU_Network)
        elif arch == 'dConv':
            instance = super().__new__(dConv_ReLU_Network)
        elif arch == 'CGNN':
            raise NotImplementedError
        else:
            raise KeyError(f"Architecture {arch} not recognised by CNN Factory.")

        params = []
        for (param_name, param) in [('kernel_sizes', kernel_sizes), ('strides', strides)]:
            try:
                if isinstance(param, int):
                    pass
                elif isinstance(param, list):
                    if not isinstance(param, np.ndarray):
                        if len(param) == len(dims) - 1:
                            param = np.array(param).astype(int)
                        elif len(param) == 1:
                            param = int(param[0][0])
                        else:
                            raise IndexError()
                elif isinstance(param, tuple):
                    if len(param) == 1:
                        param = int(param[0])
                    else:
                        raise IndexError()
                else:
                    raise TypeError()
            except (TypeError, IndexError) as error:
                raise error(f"Size mismatch between the {param_name} and layers. Expected number" \
                                                               f" of {param_name}: {len(dims) - 1}, got: {len(param)}")
            params.append(param)

        instance.__init__(dims=dims, kernel_sizes=params[0], strides=params[1], output_activation=output_activation)

        return instance

    def __init__(self):
        raise RuntimeError(f"{type(self)} is a Class Factory. Assign it to a variable. ")


class CNN_ReLU_Network(Network):

    """CNN PyTorch neural network class with ReLU hidden activations

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], strides: Union[Iterable[int], int],
                 output_activation: nn.Module = None):
            
        self.channels = np.array(dims)[:, 0]
        self.img_dims = np.array(dims)[:, 1:]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        img_out = self.img_dims[1:]
        img_in = self.img_dims[0:-1]
        padding = np.ceil((0.5 * (self.strides * (img_out - 1) - img_in + (self.kernel_sizes - 1) + 1)))
        self.padding = np.clip(padding, 0, None).astype(int)
        img_out_check = (img_in + 2 * self.padding - (self.kernel_sizes - 1) - 1) / self.strides + 1
        assert (img_out_check % 1 == 0).all(), f"Incompatible even/odd status of parameters. \\ " \
                                               f"img_out odd for (k odd, s even) or (k even, s odd, img_in even) \\" \
                                               f" img_out even for (k even, s even) or (k odd, s odd, img_in even) \\" \
                                               f" check: {img_out_check} "
        assert (img_out == img_out_check).all(), f"Failed dimensionality consistency check on CNN network. " \
                                                 f"Check value:{img_out_check}. If check_value > img_out, " \
                                                 f"img_out is impossible to reach with current (k,s)"
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
            if isinstance(self.strides, int):
                stride = self.strides
            else:
                stride = self.strides[i]
            mods.append(nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size, stride, self.padding[i]))
            if i != (len(dims) - 2): #do not append ReLU layer on the last one.
                mods.append(nn.ReLU())

        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)


class Same_CNN_ReLU_Network(Network):

    """CNN neural network Pytorch implementation with Keras style architecture to determine padding (See Conv2DSame
    class).

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers

    img_dim_out = (img_dim_in + 2*p - dil*(k-1) - 1)/s + 1
    with Conv2DSame: solve for p such that img_dim_out = img_dim_in
    """

    def __init__(self, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], strides: Union[Iterable[int], int],
                 output_activation: nn.Module = None):

        self.channels = np.array(dims)[:, 0]
        self.img_dims = np.array(dims)[:, 1:]
        self.kernel_sizes = kernel_sizes
        self.strides = strides

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


class dConv_ReLU_Network(Network):

    """dConv PyTorch neural network class with ReLU hidden activations

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    Dout = (Din−1)×stride−2×padding+dilation×(kernel_size−1) + output_padding + 1
    """

    def __init__(self, dims: Iterable[int], kernel_sizes: Union[Iterable[int], int], strides: Union[Iterable[int], int],
                 output_activation: nn.Module = None):

        self.channels = np.array(dims)[:, 0]
        self.img_dims = np.array(dims)[:, 1:]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        img_out = self.img_dims[1:]
        img_in = self.img_dims[0:-1]
        padding = 0.5 * (-img_out + (img_in - 1) * self.strides + (self.kernel_sizes - 1) + 1)
        self.out_padding = (np.mod(padding, 1) != 0).astype(int) #always 1 or 0, to match desired output size
        self.padding = np.ceil(padding).astype(int)
        assert (img_out == (img_in - 1) * self.strides - 2 * self.padding + (self.kernel_sizes - 1)
                + self.out_padding + 1).all(), "Failed dimensionality consistency check on dConv network"

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
            if isinstance(self.strides, int):
                stride = self.strides
            else:
                stride = self.strides[i]
            mods.append(nn.ConvTranspose2d(self.channels[i], self.channels[i+1], kernel_size, stride,
                                           output_padding=self.out_padding[i], padding=self.padding[i]))
            if i != (len(dims) - 2): #do not append ReLU layer on the last one.
                mods.append(nn.ReLU())

        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)