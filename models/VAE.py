import torch
import torch.nn as nn
import torch.optim as optim


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
        layers = []
        for i in range(len(enc_layer_dims) - 2):
            layers.append(nn.Linear(enc_layer_dims[i],
                                    enc_layer_dims[i + 1]))
            # Use a non-linearity
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        # Create separate final layers for each parameter (mean and log-variance)
        # We use log-variance to unconstrain the optimisation of the positive-only variance parameters
        self.mean = nn.Linear(enc_layer_dims[-2],
                              enc_layer_dims[-1])
        self.logvar = nn.Linear(enc_layer_dims[-2],
                                enc_layer_dims[-1])

    @staticmethod
    def add_model_args(parser):
        """Here we define the arguments for our encoder model."""
        parser.add_argument('--enc_layer_dims', type=int, nargs='+',
                            help='Encoder layer dimensions.')
        return parser

    def forward(self, X):
        """
        Predicts the parameters of the variational distribution

        Args:
            X (Tensor):      data, a batch of shape (B, D)

        Returns:
            mean (Tensor):   means of the variational distributions, shape (B, K)
            logvar (Tensor): log-variances of the diagonal Gaussian variational distribution, shape (B, K)
        """
        features = self.model(X)
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


class Decoder(nn.Module):
    """
    Decoder or generative network that computes the parameters of the likelihood.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        dec_layer_dims = self.hparams.dec_layer_dims
        self.data_dist = self.hparams.data_distribution

        # Create all layers except last
        layers = []
        for i in range(len(dec_layer_dims) - 2):
            layers.append(nn.Linear(dec_layer_dims[i],
                                    dec_layer_dims[i + 1]))
            # Add non-linearity
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        # Create final layers that predicts the parameters of the continuous Bernoulli
        self.logits = nn.Linear(dec_layer_dims[-2],
                                dec_layer_dims[-1])

    @staticmethod
    def add_model_args(parser):
        """Here we define the arguments for our decoder model."""
        parser.add_argument('--dec_layer_dims', type=int, nargs='+',
                            help='Decoder layer dimensions.')
        return parser

    def forward(self, Z):
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K)

        Returns:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (M, B, D)
        """
        features = self.model(Z)
        logits = self.logits(features)

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
            log_px_given_z = evaluate_logprob_continuous_bernoulli(X, logits=logits)
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
        Returns the success probability p of the continuous Bernoulli

        Args:
            logits (Tensor):   parameters of the continuous Bernoulli, shape (*, B, D)

        Returns:
            p (Tensor):  success probability p of the continuous Bernoulli, shape (*, B, D)
        """
        if self.data_dist == 'Bernoulli':
            dist = torch.distributions.Bernoulli(logits=logits)
        elif self.data_dist == 'ContinuousBernoulli':
            dist = torch.distributions.ContinuousBernoulli(logits=logits)
        else:
            raise NotImplementedError("Specified Data Distribution Invalid or Not Currently Implemented")
        return dist.probs


class VAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Use the encoder and decoder implemented above
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

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
                            help='The data distribution familly of the Decoder.')

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
