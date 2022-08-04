import torch


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


def evaluate_logprob_one_hot_categorical(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B,), entries are int representing category labels
        logits (Tensor): parameters of the C-categories OneHotCategorical distribution,
                         a batch of shape (B, C)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.OneHotCategorical(logits=logits)
    return cb.log_prob(X)


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