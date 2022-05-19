import torch
import torch.nn as nn
import torch.optim as optim
from util import BinaryTransform, layer_dim
from models.networks import FC_ReLU_Network, CNN_ReLU_Network, dConv_ReLU_Network
from models.layers import Reshape

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
    mean, logvar = encoder(X)  # (B, H), (B, H)

    # Sample the latents using the reparametrisation trick
    Z = sample_gaussian_with_reparametrisation(
        mean, logvar, num_samples=num_samples)  # (M, B, H)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)

    # logits = torch.randn(num_samples, *X.shape) for testing
    logits = decoder(Z)  # (M, B, D)

    # Compute KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, logvar)  # (B,)

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)

    if data_dist == 'Bernoulli':
        neg_cross_entropy = evaluate_logprob_bernoulli(X, logits=logits).mean(dim=0)  # (B,)
    elif data_dist == 'ContinuousBernoulli':
        neg_cross_entropy = evaluate_logprob_continuous_bernoulli(X, logits=logits).mean(dim=0)  # (B,)
    else:
        raise NotImplementedError(f"Specified Data Distribution {data_dist} Invalid or Not Currently Implemented")

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
        self.mean = nn.Linear(*enc_layer_dims[-2],
                              *enc_layer_dims[-1])
        self.logvar = nn.Linear(*enc_layer_dims[-2],
                                *enc_layer_dims[-1])

    def create_model(self, dims: Iterable[int]):
        raise NotImplementedError

    @staticmethod
    def add_model_args(parser):
        """Here we define the arguments for our encoder model."""
        parser.add_argument('--enc_architecture', type=str,
                            choices=['FC', 'CNN'],
                            help='Layer Architecture of Encoder Model')

        return parser

    @staticmethod
    def add_extra_args(parser):
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
        features = X
        for net in self.model:
            features = net(features)
        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar

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
        return [FC_ReLU_Network(dims, output_activation=nn.modules.activation.ReLU)]

    @staticmethod
    def add_extra_args(parser):
        parser.add_argument('--enc_layer_dims', type=int, nargs='+',
                            help='Encoder layer dimensions.')
        return parser


class CNNEncoder(Encoder):

    def __init__(self, hparams):
        self.kernel_sizes = hparams.enc_kernel_size[0] #TODO: handle lists of ints too
        super().__init__(hparams)

    def create_model(self, dims):
        cnn_layer_inds = [len(dim)==3 for dim in dims]
        cnn_layer_end = cnn_layer_inds.index(False)
        self.cnn_net = CNN_ReLU_Network(dims[:cnn_layer_end], kernel_sizes=self.kernel_sizes, output_activation=nn.ReLU)
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

        X = X.reshape(X.shape[0], *self.hparams.enc_layer_dims[0]) #(B, D) -> (B, C, H, W)
        #X = torch.permute(X, (0, 3, 1, 2)) #(B, H, W, C) -> (B, C, H, W) TODO: work out how to permute for mazes

        features = X
        for net in self.model:
            features = net(features)

        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar

    @staticmethod
    def add_extra_args(parser):
        """Here we define the arguments for our encoder model."""
        parser.add_argument('--enc_layer_dims', type=layer_dim, nargs='+',
                            help='Encoder layer dimensions.')
        parser.add_argument('--enc_kernel_size', type=int, nargs='+',
                            help='Encoder Kernel size.')
        return parser


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

    @staticmethod
    def add_model_args(parser):
        """Here we define the arguments for our decoder model."""
        parser.add_argument('--dec_architecture', type=str,
                            choices=['FC', 'dConv'],
                            help='Layer Architecture of Decoder Model')

        return parser

    @staticmethod
    def add_extra_args(parser):
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
        return [FC_ReLU_Network(dims, output_activation=None),]

    @staticmethod
    def add_extra_args(parser):
        parser.add_argument('--dec_layer_dims', type=int, nargs='+',
                            help='Decoder layer dimensions.')
        return parser


class dConvDecoder(Decoder):

    def __init__(self, hparams):
        self.kernel_sizes = hparams.dec_kernel_size[0] #TODO: handle lists of ints too
        super().__init__(hparams)

    def create_model(self, dims):
        fc_layer_inds = [len(dim)==1 for dim in dims]
        fc_layer_end = fc_layer_inds.index(False)
        cnn_layer_inds = list(np.array([(len(dims[i])==3 and dims[i][1]==dims[i+1][1]) for i in range(0,len(dims)-1)]))
        try:
            cnn_layer_start = cnn_layer_inds.index(True)
        except ValueError:
            cnn_layer_start = None
        self.bottleneck = FC_ReLU_Network(dims[:fc_layer_end+1], output_activation=nn.ReLU)
        self.lin2deconv = Reshape(-1, *dims[fc_layer_end])
        if cnn_layer_start == fc_layer_end:
            self.dconv_layer = None
        else:
            self.dconv_layer = dConv_ReLU_Network(dims[fc_layer_end:cnn_layer_start+1], kernel_sizes=self.kernel_sizes, output_activation=nn.ReLU)

        if cnn_layer_start is None:
            self.conv_layer = None
        else:
            self.conv_layer = CNN_ReLU_Network(dims[cnn_layer_start:], kernel_sizes=self.kernel_sizes, output_activation=None)

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

        logits = torch.permute(logits, (0, 2, 3, 1)) #(M * B, C, H, W) => (M * B, H, W, C)
        logits = logits.reshape(logits.shape[0], -1) # (M * B, H, W, C) => (M * B, H * W * C)

        # (M * B, H * W * C) => (M, B, H * W * C)
        if reshape:
            logits = logits.reshape(M, B, logits.shape[-1])

        return logits

    @staticmethod
    def add_extra_args(parser):
        """Here we define the arguments for our decoder model."""
        parser.add_argument('--dec_layer_dims', type=layer_dim, nargs='+',
                            help='Decoder layer dimensions.')
        parser.add_argument('--dec_kernel_size', type=int, nargs='+',
                            help='Decoder Kernel size.')
        return parser


class VAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Use the encoder and decoder implemented above
        # TODO: check again if best way after implementing the CNN arch. Consider moving into Encoder class.
        if self.hparams.enc_architecture == 'FC':
            self.encoder = FCEncoder(hparams)
        elif self.hparams.enc_architecture == 'CNN':
            self.encoder = CNNEncoder(hparams)

        if self.hparams.dec_architecture == 'FC':
            self.decoder = FCDecoder(hparams)
        elif self.hparams.dec_architecture == 'dConv':
            self.decoder = dConvDecoder(hparams)
    @staticmethod
    def add_model_args(parser):
        """Here we define the arguments for our decoder model."""
        parser = Encoder.add_model_args(parser)
        parser = Decoder.add_model_args(parser)

        parser.add_argument('--gradient_type', type=str,
                            choices=['pathwise', 'score'],
                            help='Variational model gradient estimation method.')
        parser.add_argument('--num_variational_samples',
                            type=int, default=1,
                            help=('The number of samples from the variational '
                                  'distribution to approximate the expectation.'))
        parser.add_argument('--data_distribution',
                            type=str, choices=['Bernoulli', 'ContinuousBernoulli'],
                            help='The data distribution family of the Decoder.')

        return parser

    def forward(self, X):
        """
        Computes the variational ELBO

        Args:
            X (Tensor):  data, a batch of shape (B, K)

        Returns:
            elbos (Tensor): per data-point elbos, shape (B, D)
        """
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

    def elbo_with_score_function_gradients(self, X):
        # Reuse the implemented code
        return elbo_with_score_function_gradients(X, encoder=self.encoder, decoder=self.decoder,
                                                  num_samples=self.hparams.num_variational_samples,
                                                  data_dist=self.hparams.data_distribution)
