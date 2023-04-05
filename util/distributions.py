import torch
import numpy as np
from omegaconf import DictConfig

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
    return cb.log_prob(X).mean(dim=-1)


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


def evaluate_logprob_diagonal_gaussian(Z, *, mean, std):
    """
    Evaluates log-probability of the diagonal Gaussian distribution

    Args:
        Z (Tensor):      latent vectors, a batch of shape (*, B, H)
        mean (Tensor):   mean of diagonal Gaussian, a batch of shape (*, B, H)
        std (Tensor):    std of diagonal Gaussian, a batch of shape (*, B, H)

    Returns:
        logqz (Tensor): log-probabilities of the inputs Z, a batch of shape (*, B)
                        where `*` corresponds to any additional dimensions of the input arguments,
                        for example a dimension representing the samples used to approximate
                        the expectation in the ELBO
    """
    gauss = torch.distributions.Normal(loc=mean, scale=std)
    return gauss.log_prob(Z).sum(dim=-1)


def compute_kld_with_standard_gaussian(q_mean, q_std):
    """
    Computes KL(q(z|x)||p(z)) between the variational diagonal
    Gaussian distribution q and standard Gaussian prior p

    Args:
        q_mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, H)
        q_std (Tensor):    std of diagonal Gaussian q, a batch of shape (B, H)

    Returns:
        kld (Tensor): KL divergences between q and p, a batch of shape (B,)
    """

    q_var = q_std**2
    q_logvar = q_var.log()

    kld = -0.5 * (1 + q_logvar - q_mean ** 2 - q_var).mean(dim=-1)

    return kld


def sample_gaussian_with_reparametrisation(mean, std, *, num_samples=1):
    """
    Samples the Gaussian distribution using the reparametrisation trick

    Args:
        mean (Tensor):     mean of diagonal Gaussian q, a batch of shape (B, K)
        std (Tensor):      std of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):        Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """

    eps = torch.randn(num_samples, *mean.shape, dtype=std.dtype, device=std.device)

    Z = mean + eps * std

    # Making sure that the variational samples are on the same device as the input argument (i.e. CPU or GPU)
    return Z.to(device=std.device)


def sample_gaussian_without_reparametrisation(mean, std, *, num_samples=1):
    """
    Samples the Gaussian distribution without attaching then to the computation graph

    Args:
        mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, K)
        std (Tensor): std of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):      Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """
    return sample_gaussian_with_reparametrisation(mean, std, num_samples=num_samples).detach()


def compute_weights(scores: np.ndarray, params: DictConfig)-> np.ndarray:
    if params.distribution == 'power':
        weights = (scores.clip(0)) ** (1. / params.temperature)
    elif params.distribution == 'power_rank':
        temp = np.flip(scores.argsort())
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp)) + 1
        weights = 1 / ranks ** (1. / params.temperature)
    else:
        raise ValueError(f"Unknown distribution: {params.distribution}")

    z = np.sum(weights)
    if z > 0:
        weights /= z
    else:
        weights = np.ones_like(weights, dtype=np.float) / len(weights)
        weights /= np.sum(weights)

    return weights
