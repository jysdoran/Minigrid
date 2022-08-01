import dgl
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from util import BinaryTransform
from models.networks import FC_ReLU_Network, CNN_Factory, Network
from models.gnn_networks import GIN
from models.layers import Reshape
from data_generators import Batch

from typing import Iterable, Tuple
from math import prod
import numpy as np

def evaluate_logprob_continuous_bernoulli(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, D)
        logits (Tensor): parameters of the continuous Bernoulli,
                         a batch of shape (B, D)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.ContinuousBernoulli(logits=logits)
    return cb.log_prob(X).sum(dim=-1)

def evaluate_logprob_bernoulli(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, D)
        logits (Tensor): parameters of the continuous Bernoulli,
                         a batch of shape (B, D)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.Bernoulli(logits=logits)
    return cb.log_prob(X).sum(dim=-1)

def evaluate_logprob_diagonal_gaussian(Z, *, mean, logvar):
    """
    Evaluates log-probability of the diagonal Gaussian distribution

    Args:
        Z (Tensor):      latent vectors, a batch of shape (*, B, H)
        mean (Tensor):   mean of diagonal Gaussian, a batch of shape (*, B, H)
        logvar (Tensor): log-variance of diagonal Gaussian, a batch of shape (*, B, H)

    Returns:
        logqz (Tensor): log-probabilities of the inputs Z, a batch of shape (*, B)
                        where `*` corresponds to any additional dimensions of the input arguments,
                        for example a dimension representing the samples used to approximate
                        the expectation in the ELBO
    """
    gauss = torch.distributions.Normal(loc=mean, scale=torch.exp(0.5*logvar))
    return gauss.log_prob(Z).sum(dim=-1)


def compute_kld_with_standard_gaussian(q_mean, q_logvar):
    """
    Computes KL(q(z|x)||p(z)) between the variational diagonal
    Gaussian distribution q and standard Gaussian prior p

    Args:
        q_mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, H)
        q_logvar (Tensor): log-variance of diagonal Gaussian q, a batch of shape (B, H)

    Returns:
        kld (Tensor): KL divergences between q and p, a batch of shape (B,)
    """

    q_var = q_logvar.exp()

    kld = -0.5 * (1 + q_logvar - q_mean ** 2 - q_var).sum(dim=-1)

    return kld


def sample_gaussian_with_reparametrisation(mean, logvar, *, num_samples=1):
    """
    Samples the Gaussian distribution using the reparametrisation trick

    Args:
        mean (Tensor):     mean of diagonal Gaussian q, a batch of shape (B, K)
        logvar (Tensor):   log-variance of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):        Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """

    std = torch.exp(0.5 * logvar)
    # eps = torch.tensor(np.random.randn(num_samples, *mean.shape), dtype=std.dtype, device=std.device)
    eps = torch.randn(num_samples, *mean.shape, dtype=std.dtype, device=std.device)  # more efficient

    Z = mean + eps * std

    # Making sure that the variational samples are on the same device as the input argument (i.e. CPU or GPU)
    return Z.to(device=std.device)


def elbo_with_pathwise_gradients(X, *, encoder, decoder, num_samples, data_dist=None):
    """
    Evaluate the ELBO for each data-point in X

    Args:
        X (Tensor):                 data, a batch of shape (B, D)
        decoder (torch.nn.Module):  is a neural network that provides the parameters
                                    of the generative model p(x|z; θ)
        encoder (torch.nn.Module):  is a neural network that provides the parameters
                                    of the variational distribution q(z|x; φ).
        num_samples (int):          the number of samples used in the Monte Carlo averaging

    Returns:
        elbos (Tensor):             the ELBOs for each input X, a tensor of shape (B,)
    """
    # Notes:
    # - Variational distribution <=> q(z|x) (parametrised by Encoder : mean, logvar)
    # - Generative Model p(x|z) (parametrised by Decoder: logits)
    # - M <=> num_samples- B <=> minibatch size
    # - D <=> dimension of X
    # - H <=> dimension of Z
    # - logits <=> theta (parameters of p(x|z))
    # - mean, logvar <=> phi (parameters of q(z|x))
    # - Z ~ q(z|x), sampled with reparametrisation
    # - X ~ p(x|z) (generated X | Z) <=> N/A here (but would be cb.sample())
    #
    # Important to understand:
    # Encoder output is the parameters of the Gaussian q(z|x) at that specific X
    # Decoder output is the parameters of the Continuous Bernouilli p(x|z) at that specific Z

    # Evaluate the encoder network to obtain the parameters of the
    # variational distribution (i.e q(z|x))

    # mean = torch.randn(X.shape[0], 10) for testing
    # logvar = torch.randn(*mean.shape) for testing

    reshape = False
    if len(X.shape) > 2: # X is of dim (B, C, H, W) or (B, H, W)
        reshape = True

    mean, logvar = encoder(X)  # (B, H), (B, H)

    # Sample the latents using the reparametrisation trick
    Z = sample_gaussian_with_reparametrisation(
        mean, logvar, num_samples=num_samples)  # (M, B, H)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)

    # logits = torch.randn(num_samples, *X.shape) for testing
    logits = decoder(Z)  # (M, B, C, H, W)

    # Compute KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, logvar)  # (B,)

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)

    if reshape:
        X = X.flatten(1) # (B, d1, d2, ..., dn) -> (B, D)
        logits = logits.flatten(2) # (M, B, d1, d2, ..., dn) -> (M, B, D)

    if data_dist == 'Bernoulli':
        neg_cross_entropy = evaluate_logprob_bernoulli(X, logits=logits).mean(dim=0)  # (B,)
    elif data_dist == 'ContinuousBernoulli':
        neg_cross_entropy = evaluate_logprob_continuous_bernoulli(X, logits=logits).mean(dim=0)  # (B,)
    else:
        raise NotImplementedError(f"Specified Data Distribution {data_dist} Invalid or Not Currently Implemented")

    # ELBO for each data-point
    elbos = neg_cross_entropy - kld  # (B,)

    return elbos

def elbo_with_pathwise_gradients_gnn(X, *, encoder, decoder, num_samples, data_dist=None, permutations=None):

    mean, logvar = encoder(X)  # (B, H), (B, H)

    # Sample the latents using the reparametrisation trick
    Z = sample_gaussian_with_reparametrisation(
        mean, logvar, num_samples=num_samples)  # (M, B, H)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)

    # logits = torch.randn(num_samples, *X.shape) for testing
    logits = decoder(Z)  # (M, B, n_nodes-1, 2)

    graphs = dgl.unbatch(X)
    #TODO: find a way to do this efficiently on GPU, maybe convert logits to sparse tensor
    n_nodes = graphs[0].num_nodes()
    A_in = torch.empty((len(graphs), n_nodes, n_nodes)).to(logits.device)
    for m in range(len(graphs)):
        A_in[m] = graphs[m].adj().to_dense()
    if permutations is not None:
        permutations = permutations.to(logits.device)
        A_in = [A_in[:,permutations[i]][:,:, permutations[i]] for i in range(permutations.shape[0])]
        A_in = torch.stack(A_in, dim=1).to(logits.device)  # B, P, N, N
        A_in = A_in.reshape(A_in.shape[0]*A_in.shape[1],*A_in.shape[2:]) #B*P, N, N
        A_in = Batch.encode_adj_to_reduced_adj(A_in) #B*P, n_nodes - 1, 2
        logits = logits.repeat_interleave(permutations.shape[0], dim=1) # (M, B, n_nodes-1, 2) -> (M, B*P, n_nodes-1, 2)

    # Compute KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, logvar)  # (B,)

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)

    A_in = A_in.reshape(A_in.shape[0], -1) # (B | B * P, d1, d2, ..., dn) -> (B | B*P, D=2*(n_nodes - 1))
    logits = logits.reshape(*logits.shape[0:2], -1) # (M, B, n_nodes-1, 2) -> (M, B, D=2*(n_nodes - 1))

    if data_dist == 'Bernoulli':
        neg_cross_entropy = evaluate_logprob_bernoulli(A_in, logits=logits).mean(dim=0)  # (B,) | (B*P,)
    elif data_dist == 'ContinuousBernoulli':
        neg_cross_entropy = evaluate_logprob_continuous_bernoulli(A_in, logits=logits).mean(dim=0)  # (B,) | (B*P,)
    else:
        raise NotImplementedError(f"Specified Data Distribution {data_dist} Invalid or Not Currently Implemented")

    if permutations is not None:
        neg_cross_entropy = neg_cross_entropy.reshape(-1, permutations.shape[0]) # (B*P,) -> (B,P)
        neg_cross_entropy, _ = torch.max(neg_cross_entropy, dim=1) # (B,P,) -> (B,)
    # ELBO for each data-point
    elbos = neg_cross_entropy - kld  # (B,)

    return elbos

def sample_gaussian_without_reparametrisation(mean, logvar, *, num_samples=1):
    """
    Samples the Gaussian distribution without attaching then to the computation graph

    Args:
        mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, K)
        logvar (Tensor): log-variance of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):      Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """
    return sample_gaussian_with_reparametrisation(mean, logvar, num_samples=num_samples).detach()


def elbo_with_score_function_gradients(X, *, encoder, decoder, num_samples, data_dist=None):
    """
    Evaluate the surrogate ELBO for each data-point in X. The surrogate
    elbo uses score function gradient estimator to estimate the gradients
    of the variational distribution.

    Args:
        X (Tensor):                 data, a batch of shape (B, D)
        decoder (torch.nn.Module):  is a neural network that provides the parameters
                                    of the generative model p(x|z; θ)
        encoder (torch.nn.Module):  is a neural network that provides the parameters
                                    of the variational distribution q(z|x; φ).
        num_samples (int):          the number of samples used in the Monte Carlo averaging

    Returns:
        elbos (Tensor):             the ELBOs for each input X, a tensor of shape (B,)
    """
    # Evaluate the encoder network to obtain the parameters of the
    # variational distribution
    mean, logvar = encoder(X)  # (B, H), (B, H)

    # Sample the latents _without_ the reparametrisation trick
    Z = sample_gaussian_without_reparametrisation(
        mean, logvar, num_samples=num_samples)  # (M, B, H)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)
    logits = decoder(Z)  # (M, B, D)

    #
    # ELBO
    #

    # KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, logvar)

    # log p(x | z)
    if data_dist == 'Bernoulli':
        log_px_given_z = evaluate_logprob_bernoulli(X, logits=logits)  # (B,)
    elif data_dist == 'ContinuousBernoulli':
        log_px_given_z = evaluate_logprob_continuous_bernoulli(X, logits=logits)  # (B,)
    else:
        raise NotImplementedError(f"Specified Data Distribution {data_dist} Invalid or Not Currently Implemented")

    # ~E_{q(z|x)}[ p(x | z) ]
    neg_cross_entropy = log_px_given_z.mean(dim=0)

    # Per-data-point ELBO that does not estimate the gradients of the expectation parameters
    elbos = neg_cross_entropy - kld  # (B,)

    #
    # Score function surrogate
    #

    # log q(z | x)
    log_qz_given_x = evaluate_logprob_diagonal_gaussian(Z, mean=mean, logvar=logvar)

    # Surrogate loss for score function gradient estimator of the expectation parameters
    surrogate_loss = (log_qz_given_x * log_px_given_z.detach()).mean(dim=0)

    # ELBO with score function gradients
    # Note: We do this to still compute the loss, while getting the gradients of the surrogate_loss.
    surrogate_elbos = elbos + surrogate_loss - surrogate_loss.detach()  # (B,)

    ###
    return surrogate_elbos


class Encoder(nn.Module):
    """
    Encoder or inference network that predicts the parameters of the variational distribution q(z|x).
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        enc_layer_dims = self.hparams.enc_layer_dims

        # Create all layers except last
        self.model = self.create_model(enc_layer_dims[:-1])

        # Create separate final layers for each parameter (mean and log-variance)
        # We use log-variance to unconstrain the optimisation of the positive-only variance parameters

        #TODO: handle GNN more graciously
        if isinstance(enc_layer_dims[-2], int):
            bneck_in = (enc_layer_dims[-2],)
        else:
            bneck_in = enc_layer_dims[-2]
        if isinstance(enc_layer_dims[-1], int):
            bneck_out = (enc_layer_dims[-1],)
        else:
            bneck_out = enc_layer_dims[-1]

        self.mean = nn.Linear(*bneck_in, *bneck_out)
        self.logvar = nn.Linear(*bneck_in, *bneck_out)

    def create_model(self, dims: Iterable[int]):
        raise NotImplementedError

    def forward(self, X):
        """
        Predicts the parameters of the variational distribution

        Args:
            X (Tensor):      data, a batch of shape (B, D)

        Returns:
            mean (Tensor):   means of the variational distributions, shape (B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (B, K)
        """
        raise NotImplementedError

    def sample_with_reparametrisation(self, mean, logvar, *, num_samples=1):
        # Reuse the implemented code
        return sample_gaussian_with_reparametrisation(mean, logvar, num_samples=num_samples)

    def sample_without_reparametrisation(self, mean, logvar, *, num_samples=1):
        # Reuse the implemented code
        return sample_gaussian_without_reparametrisation(mean, logvar, num_samples=num_samples)

    def log_prob(self, mean, logvar, Z):
        """
        Evaluates the log_probability of Z given the parameters of the diagonal Gaussian

        Args:
            mean (Tensor):   means of the variational distributions, shape (*, B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (*, B, K)
            Z (Tensor):      latent vectors, shape (*, B, K)

        Returns:
            logqz (Tensor):  log-probability of Z, a batch of shape (*, B)
        """
        # Reuse the implemented code
        return evaluate_logprob_diagonal_gaussian(Z, mean=mean, logvar=logvar)

class FCEncoder(Encoder):

    def __init__(self, hparams):
        super().__init__(hparams)

    def create_model(self, dims):
        return nn.ModuleList([FC_ReLU_Network(dims, output_activation=nn.modules.activation.ReLU)])

    def forward(self, X):
        """
        Predicts the parameters of the variational distribution

        Args:
            X (Tensor):      data, a batch of shape (B, D)

        Returns:
            mean (Tensor):   means of the variational distributions, shape (B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (B, K)
        """
        features = X.flatten(1)
        for net in self.model:
            features = net(features)
        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar


class CNNEncoder(Encoder):

    def __init__(self, hparams):
        super().__init__(hparams)

    def create_model(self, dims):
        cnn_layer_inds = [len(dim)==3 for dim in dims]
        cnn_layer_end = cnn_layer_inds.index(False)
        self.cnn_net = CNN_Factory(dims[:cnn_layer_end], self.hparams.enc_kernel_size, self.hparams.enc_strides,
                                   output_activation=nn.ReLU, arch='CNN', same_padding=self.hparams.same_padding)
        #self.cnn_net = CNN_ReLU_Network(dims[:cnn_layer_end], kernel_sizes=self.kernel_sizes, output_activation=nn.ReLU)
        self.flatten_layer = nn.Flatten()
        self.lin_layer = FC_ReLU_Network(dims[cnn_layer_end-1:], output_activation=nn.ReLU)

        model = [self.cnn_net, self.flatten_layer, self.lin_layer]
        model = list(filter(None, model))

        return model

    def forward(self, X):
        """
        Predicts the parameters of the variational distribution

        Args:
            X (Tensor):      data, a batch of shape (B, D)

        Returns:
            mean (Tensor):   means of the variational distributions, shape (B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (B, K)
        """

        X = X.reshape(X.shape[0], *self.hparams.enc_layer_dims[0]) #(B, D) -> (B, C, H, W) #however already handled by default
        features = X
        for net in self.model:
            features = net(features)

        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar


class GNNEncoder(Encoder):

    def __init__(self, hparams):
        super().__init__(hparams)

    def create_model(self, dims): #TODO: actually use dims
        #try dropout 0, #TODO: fix input dim
        self.gnn_net = GIN(num_layers=self.hparams.enc_convolutions, num_mlp_layers=2, input_dim=dims[0], hidden_dim=dims[1],
                 output_dim=dims[1], final_dropout=0, learn_eps=False, graph_pooling_type='mean',
                 neighbor_pooling_type='sum', n_nodes=169)
        self.flatten_layer = nn.Flatten()
        self.linear = FC_ReLU_Network([self.gnn_net.output_dim, *dims[2:]], output_activation=nn.ReLU)
        model = [self.gnn_net, self.flatten_layer, self.linear]
        model = list(filter(None, model))

        return model

    def forward(self, X):
        """
        Predicts the parameters of the variational distribution

        Args:
            X (Tensor):      data, a batch of shape (B, D)

        Returns:
            mean (Tensor):   means of the variational distributions, shape (B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (B, K)
        """

        graph = X
        features = graph.ndata['feat'].float()
        features = self.gnn_net(graph, features)

        for net in self.model[1:]:
            features = net(features)

        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar



class Decoder(nn.Module):
    """
    Decoder or generative network that computes the parameters of the likelihood.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_dist = self.hparams.data_distribution
        self.model = self.create_model(self.hparams.dec_layer_dims)

    def create_model(self, dims: Iterable):
        raise NotImplementedError

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K)

        Returns:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (M, B, D)
        """
        logits = Z
        for net in self.model:
            logits = net(logits)

        return logits

    def log_prob(self, logits, X):
        """
        Evaluates the log_probability of X given the parameters of the continuous Bernoulli

        Args:
            logits (Tensor): parameters of the continuous Bernoulli, shape (*, B, D)
            X (Tensor):      data, shape (*, B, D)

        Returns:
            logpx (Tensor):  log-probability of X, a batch of shape (*, B)
        """
        # Reuse the implemented code
        if self.data_dist == 'Bernoulli':
            return evaluate_logprob_bernoulli(X, logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            return evaluate_logprob_continuous_bernoulli(X, logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")

    # Some extra methods for analysis

    def sample(self, logits, *, num_samples=1):
        """
        Samples the continuous Bernoulli

        Args:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (*, B, D)
            num_samples (int): number of samples

        Returns:
            X (Tensor):  samples from the distribution, shape (num_samples, *, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.sample((num_samples,))

    def mean(self, logits):
        """
        Returns the mean of the continuous Bernoulli

        Args:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (*, B, D)

        Returns:
            mean (Tensor):  means of the continuous Bernoulli, shape (*, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.mean

    def param_p(self, logits):
        """
        Returns the distribution parameters

        Args:
            logits (Tensor): decoder output (non-normalised log probabilities), shape (*, B, D)

        Returns:
            p (Tensor): parameters of the distribution, shape (*, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.probs

    def param_b(self, logits, threshold: float = 0.5):
        """
        Returns a binary transformation of the distribution parameters

        Args:
            logits (Tensor): decoder output (non-normalised log probabilities), shape (*, B, D)
            :param threshold (float): Threshold for binary transformation, [0,1]

        Returns:
            b (Tensor):  Binary transformation of the distribution parameters, shape (*, B, D)
        """

        binary_transform = BinaryTransform(threshold)
        return binary_transform(self.param_p(logits))


class FCDecoder(Decoder):

    def __init__(self, hparams):
        super().__init__(hparams)

    def create_model(self, dims: Iterable[int]):
        self.bottleneck = FC_ReLU_Network(dims[0:2], output_activation=nn.ReLU)
        self.fc_net = FC_ReLU_Network(dims[1:], output_activation=None)
        return [self.bottleneck, self.fc_net]

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K) / (B, K)

        Returns:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (M, B, D) / (B, D)
        """

        base_shape = Z.shape[:-1] # (M, B) or (B)

        logits = Z
        for net in self.model:
            logits = net(logits)

        logits = logits.reshape(*base_shape, *self.hparams.data_dims)
        return logits

class dConvDecoder(Decoder):

    def __init__(self, hparams):
        super().__init__(hparams)

    def create_model(self, dims):
        fc_layer_inds = [len(dim)==1 for dim in dims]
        fc_layer_end = fc_layer_inds.index(False)
        dims_fc = dims[0:fc_layer_end+1]
        if isinstance(self.hparams.dec_kernel_size, int):
            self.kernel_size = conv_kernel_size = dconv_kernel_size = self.hparams.dec_kernel_size
        elif isinstance(self.hparams.dec_kernel_size, tuple) and len(self.hparams.dec_kernel_size) == 1:
            self.kernel_size = conv_kernel_size = dconv_kernel_size = self.hparams.dec_kernel_size[0]
        elif isinstance(self.hparams.dec_kernel_size, list) and len(self.hparams.dec_kernel_size) == 1:
            self.kernel_size = conv_kernel_size = dconv_kernel_size = self.hparams.dec_kernel_size[0][0]
        else:
            self.kernel_size, conv_kernel_size, dconv_kernel_size = None, None, None

        if isinstance(self.hparams.dec_stride, int):
            self.stride = conv_stride = dconv_stride = self.hparams.dec_stride
        elif isinstance(self.hparams.dec_stride, tuple) and len(self.hparams.dec_stride) == 1:
            self.stride = conv_stride = dconv_stride = self.hparams.dec_stride[0]
        elif isinstance(self.hparams.dec_stride, list) and len(self.hparams.dec_stride) == 1:
            self.stride = conv_stride = dconv_stride = self.hparams.dec_stride[0][0]
        else:
            self.stride, conv_stride, dconv_stride = None, None, None
        if self.hparams.dec_cnn_last_layer:
            dims_dconv = dims[fc_layer_end:-1]
            dims_conv = dims[-2:]
            output_activation_dconv = nn.ReLU
            if self.kernel_size is None:
                self.kernel_size = self.hparams.dec_kernel_size
                dconv_kernel_size = self.hparams.dec_kernel_size[:-1]
                conv_kernel_size = self.hparams.dec_kernel_size[-1]
            if self.stride is None:
                self.stride = self.hparams.dec_stride
                dconv_stride = self.hparams.dec_stride[:-1]
                conv_stride = self.hparams.dec_stride[-1]
        else:
            dims_dconv = dims[fc_layer_end:]
            if self.kernel_size is None:
                self.kernel_size = dconv_kernel_size = self.hparams.dec_kernel_size
            if self.stride is None:
                self.stride = dconv_stride = self.hparams.dec_stride
            dims_conv = None
            output_activation_dconv = None
        #cnn_layer_inds = list(np.array([(len(dims[i])==3 and dims[i][1]==dims[i+1][1]) for i in range(0,len(dims)-1)]))
        # try:
        #     cnn_layer_start = cnn_layer_inds.index(True)
        # except ValueError:
        #     cnn_layer_start = None
        self.bottleneck = FC_ReLU_Network(dims_fc, output_activation=nn.ReLU)
        self.lin2deconv = Reshape(-1, *dims[fc_layer_end])
        self.dconv_layer = CNN_Factory(dims_dconv, kernel_sizes=dconv_kernel_size,
                                       strides=dconv_stride, output_activation=output_activation_dconv, arch='dConv', same_padding=False)
        if dims_conv is None:
            self.conv_layer = None
        else:
            self.conv_layer = CNN_Factory(dims_conv, kernel_sizes=conv_kernel_size,
                                           strides=conv_stride, output_activation=None, arch='CNN', same_padding=False)

        model = [self.bottleneck, self.lin2deconv, self.dconv_layer, self.conv_layer]
        model = list(filter(None, model))

        return model

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K)

        Returns:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (M, B, D)
        """

        #(M, B, H) => (M * B, H)
        if len(Z.shape)==3:
            reshape = True
            M, B, H = Z.shape
            Z = Z.reshape(M * B, H)
        else:
            reshape = False

        logits = Z
        for net in self.model:
            logits = net(logits)

        # logits = torch.permute(logits, (0, 2, 3, 1)) #(M * B, C, H, W) => (M * B, H, W, C)
        # logits = logits.reshape(logits.shape[0], -1) # (M * B, H, W, C) => (M * B, H * W * C)

        # (M * B, H * W * C) => (M, B, H * W * C)
        if reshape:
            logits = logits.reshape(M, B, *logits.shape[1:])

        return logits

class GraphMLPDecoder(Decoder):

    def __init__(self, config, hyperparameters):
        super().__init__()
        self.config = config
        self.hparams = hyperparameters
        self.distributions = self.config.distributions
        self.model = self.create_model(*self.hparams.latent_dim, *self.hparams.decoder.layer_dim)
        if self.config.adjacency is not None:
            self.adjacency = Network((*self.hparams.decoder.layer_dim[-1], self.config.output_dim.adjacency),
                                     output_activation=nn.Sigmoid)
        else:
            self.adjacency = None
        if self.config.attributes:
            self.attribute_heads = []
            for i in range(len(self.config.attributes_names)):
                self.attribute_heads.append(Network((*self.hparams.decoder.layer_dim[-1],
                                                    *self.config.output_dim.attributes[i]),
                                                    output_activation=nn.Softmax))
            else:
                self.attribute_heads = None

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K)

        Returns:
            adj_out, f_out (Tensors):
        """
        logits = Z
        for net in self.model:
            logits = net(logits)

        if self.adjacency is not None:
            adj_out = self.adjacency(logits)
        else:
            adj_out = None

        if self.attribute_heads is not None:
            f_out = []
            for net in self.attribute_heads:
                f_out.append(net(logits))
            torch.stack(f_out)
        else:
            f_out = None

        return adj_out, f_out

    # def forward_adjacency(self, Z):
    #     logits = self._forward_model(Z)
    #     return self.adjacency(logits)
    #
    # def forward_attributes(self, Z):
    #     logits = self._forward_model(Z)
    #     f_out = []
    #     for net in self.attribute_heads:
    #         f_out.append(net(logits))
    #     f_out = torch.stack(f_out)
    #     return f_out
    #
    # def _forward_model(self, Z):
    #     """
    #     """
    #     logits = Z
    #     for net in self.model:
    #         logits = net(logits)
    #
    #     return logits

    #TODO: pickup from here. Rearrange the functions above into a distributions file?
    def log_prob(self, logits, X):
        """
        Evaluates the log_probability of X given the parameters of the continuous Bernoulli

        Args:
            logits (Tensor): parameters of the continuous Bernoulli, shape (*, B, D)
            X (Tensor):      data, shape (*, B, D)

        Returns:
            logpx (Tensor):  log-probability of X, a batch of shape (*, B)
        """
        # Reuse the implemented code
        if self.data_dist == 'Bernoulli':
            return evaluate_logprob_bernoulli(X, logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            return evaluate_logprob_continuous_bernoulli(X, logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")

    # Some extra methods for analysis

    def sample(self, logits, *, num_samples=1):
        """
        Samples the continuous Bernoulli

        Args:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (*, B, D)
            num_samples (int): number of samples

        Returns:
            X (Tensor):  samples from the distribution, shape (num_samples, *, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.sample((num_samples,))

    def mean(self, logits):
        """
        Returns the mean of the continuous Bernoulli

        Args:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (*, B, D)

        Returns:
            mean (Tensor):  means of the continuous Bernoulli, shape (*, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.mean

    def param_p(self, logits):
        """
        Returns the distribution parameters

        Args:
            logits (Tensor): decoder output (non-normalised log probabilities), shape (*, B, D)

        Returns:
            p (Tensor): parameters of the distribution, shape (*, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.probs

    def param_b(self, logits, threshold: float = 0.5):
        """
        Returns a binary transformation of the distribution parameters

        Args:
            logits (Tensor): decoder output (non-normalised log probabilities), shape (*, B, D)
            :param threshold (float): Threshold for binary transformation, [0,1]

        Returns:
            b (Tensor):  Binary transformation of the distribution parameters, shape (*, B, D)
        """

        binary_transform = BinaryTransform(threshold)
        return binary_transform(self.param_p(logits))

    def __init__(self, config):
        super().__init__(config)

    def create_model(self, dims: Iterable[int]):
        self.bottleneck = FC_ReLU_Network(dims[0:2], output_activation=nn.ReLU)
        self.fc_net = FC_ReLU_Network(dims[1:], output_activation=None)
        return [self.bottleneck, self.fc_net]

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K) / (B, K)

        Returns:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (M, B, D) / (B, D)
        """

        base_shape = Z.shape[:-1] # (M, B) or (B)

        logits = Z
        for net in self.model:
            logits = net(logits)

        logits = logits.reshape(*base_shape, *self.config.data_dims)
        return logits


class VAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Use the encoder and decoder implemented above
        if self.hparams.architecture == 'FC':
            self.encoder = FCEncoder(hparams)
            self.decoder = FCDecoder(hparams)
        elif self.hparams.architecture == 'CNN':
            self.encoder = CNNEncoder(hparams)
            self.decoder = dConvDecoder(hparams)
        elif self.hparams.architecture == 'GNN':
            self.encoder = GNNEncoder(hparams)
            self.decoder = FCDecoder(hparams)
            if self.hparams.augmented_inputs:
                transforms = torch.tensor([[[1, 0], [0, 1]],
                                               [[1, 0], [0, -1]],
                                               [[0, 1], [1, 0]],
                                               [[0, 1], [-1, 0]],
                                               [[-1, 0], [0, 1]],
                                               [[-1, 0], [0, -1]],
                                               [[0, -1], [1, 0]],
                                               [[0, -1], [-1, 0]]], dtype=torch.int)
                self.permutations = Batch.augment_adj(self.hparams.num_nodes, transforms).long()

        else:
            raise argparse.ArgumentError(f"VAE Architecture argument {self.hparams.architecture} not recognised.")

    def forward(self, X):
        """
        Computes the variational ELBO

        Args:
            X (Tensor):  data, a batch of shape (B, K)

        Returns:
            elbos (Tensor): per data-point elbos, shape (B, D)
        """
        if self.hparams.architecture == 'GNN':
            if self.hparams.gradient_type == 'pathwise':
                return self.elbo_with_pathwise_gradients_gnn(X)
            else:
                raise ValueError(f'gradient_type={self.hparams.gradient_type} is invalid')
        else:
            if self.hparams.gradient_type == 'pathwise':
                return self.elbo_with_pathwise_gradients(X)
            elif self.hparams.gradient_type == 'score':
                return self.elbo_with_score_function_gradients(X)
            else:
                raise ValueError(f'gradient_type={self.hparams.gradient_type} is invalid')


    def elbo_with_pathwise_gradients(self, X):
        # Reuse the implemented code
        return elbo_with_pathwise_gradients(X, encoder=self.encoder, decoder=self.decoder,
                                            num_samples=self.hparams.num_variational_samples,
                                            data_dist=self.hparams.data_distribution)

    def elbo_with_pathwise_gradients_gnn(self, X):
        # Reuse the implemented code
        return elbo_with_pathwise_gradients_gnn(X, encoder=self.encoder, decoder=self.decoder,
                                            num_samples=self.hparams.num_variational_samples,
                                            data_dist=self.hparams.data_distribution, permutations=self.permutations)

    def elbo_with_score_function_gradients(self, X):
        # Reuse the implemented code
        return elbo_with_score_function_gradients(X, encoder=self.encoder, decoder=self.decoder,
                                                  num_samples=self.hparams.num_variational_samples,
                                                  data_dist=self.hparams.data_distribution)

    @property
    def num_parameters(self):
        total_params = 0
        for parameters_blocks in self.parameters():
            for parameter_array in parameters_blocks:
                array_shape = [*parameter_array.shape]
                if array_shape:
                    num_params = np.prod(array_shape)
                    total_params += num_params

        return total_params


class GraphVAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = GNNEncoder(config.configuration.encoder, config.hyperparameters)
        self.decoder = GraphMLPDecoder(config.configuration.decoder, config.hyperparameters)

        if self.config.model_config.configuration.model.augmented_inputs:
            transforms = torch.tensor(self.config.configuration.model.transforms, dtype=torch.int)
            self.permutations = Batch.augment_adj(self.config.configuration.data.graph_max_nodes, transforms).long()
        else: self.permutations = None

    def forward(self, X):
        """
        Computes the variational ELBO

        Args:
            X (Tensor):  data, a batch of shape (B, K)

        Returns:
            elbos (Tensor): per data-point elbos, shape (B, D)
        """
        if self.config.configuration.model.gradient_type == 'pathwise':
            return self.elbo_with_pathwise_gradients(X)
        else:
            raise ValueError(f'gradient_type={self.hparams.gradient_type} is invalid')

    def elbo_with_pathwise_gradients(self, X):
        # Reuse the implemented code
        return elbo_with_pathwise_gradients_gnn(X, encoder=self.encoder, decoder=self.decoder,
                                            num_samples=self.config.configuration.model.num_variational_samples,
                                            data_dist=self.config.configuration.decoder.distributions,
                                            permutations=self.permutations)

    @property
    def num_parameters(self):
        total_params = 0
        for parameters_blocks in self.parameters():
            for parameter_array in parameters_blocks:
                array_shape = [*parameter_array.shape]
                if array_shape:
                    num_params = np.prod(array_shape)
                    total_params += num_params

        return total_params

