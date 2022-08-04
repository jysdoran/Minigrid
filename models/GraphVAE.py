import dgl
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

from data_generators import Batch
from models.gnn_networks import GIN
from models.networks import FC_ReLU_Network, Network
from util.distributions import sample_gaussian_with_reparametrisation, compute_kld_with_standard_gaussian, \
    evaluate_logprob_bernoulli, evaluate_logprob_one_hot_categorical, evaluate_logprob_diagonal_gaussian
from util.util import BinaryTransform


def graphVAE_elbo_pathwise(X, *, encoder, decoder, num_samples, elbo_coeffs, permutations=None):
    # TODO: permutations not functional at the moment, because not implemented for Fx

    mean, logvar = encoder(X)  # (B, H), (B, H)

    # Sample the latents using the reparametrisation trick
    Z = sample_gaussian_with_reparametrisation(
        mean, logvar, num_samples=num_samples)  # (M, B, H)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)

    # logits = torch.randn(num_samples, *X.shape) for testing
    logits_A, logits_Fx = decoder(Z)  # (M, B, n_nodes-1, 2)

    graphs = dgl.unbatch(X)
    n_nodes = decoder.shared_params.graph_max_nodes
    reconstructed_features = tuple(decoder.config.attributes_mapping)
    f_dim = len(reconstructed_features) #only the reconstructed dimensions [empty, start, goal]
    A_in = torch.empty((len(graphs), n_nodes, n_nodes)).to(logits_A)
    Fx = torch.empty((len(graphs), n_nodes, f_dim)).to(logits_Fx)

    #TODO: find a way to do this efficiently on GPU, maybe convert logits to sparse tensor
    Fx = []
    for m in range(len(graphs)):
        A_in[m] = graphs[m].adj().to_dense()
        Fx.append(graphs[m].ndata['feat'][..., reconstructed_features].to(logits_Fx))
    Fx = torch.stack(Fx, dim=0).to(logits_Fx)

    if permutations is not None:
        permutations = permutations.to(logits_A.device)
        A_in = [A_in[:,permutations[i]][:,:, permutations[i]] for i in range(permutations.shape[0])]
        A_in = torch.stack(A_in, dim=1).to(logits_A.device)  # B, P, N, N
        A_in = A_in.reshape(A_in.shape[0]*A_in.shape[1],*A_in.shape[2:]) #B*P, N, N
        A_in = Batch.encode_adj_to_reduced_adj(A_in) #B*P, n_nodes - 1, 2
        logits_A = logits_A.repeat_interleave(permutations.shape[0], dim=1) # (M, B, n_nodes-1, 2) -> (M, B*P, n_nodes-1, 2)

    # Compute KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, logvar)  # (B,)

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)

    A_in = A_in.reshape(A_in.shape[0], -1) # (B | B * P, d1, d2, ..., dn) -> (B | B*P, D=2*(n_nodes - 1))
    logits_A = logits_A.reshape(*logits_A.shape[0:2], -1) # (M, B, n_nodes-1, 2) -> (M, B, D=2*(n_nodes - 1))
    neg_cross_entropy_A = evaluate_logprob_bernoulli(A_in, logits=logits_A).mean(dim=0)  # (B,) | (B*P,)

    neg_cross_entropy_Fx = []
    for i in range(len(decoder.attribute_distributions)):
        if decoder.attribute_distributions[i] == "bernoulli":
            neg_cross_entropy_Fx.append(evaluate_logprob_bernoulli(Fx[..., i], logits=logits_Fx[..., i]).mean(dim=0)) # (B,)
        # Note: more efficient way to do this is to "bundle" start and goal within a single batch (i.e. B,D),
        # but requires modifying evaluate_logprob_one_hot_categorical()
        elif decoder.attribute_distributions[i] == "one_hot_categorical":
            neg_cross_entropy_Fx.append(evaluate_logprob_one_hot_categorical(Fx[..., i], logits=logits_Fx[..., i]).mean(dim=0))
        else:
            raise NotImplementedError(f"Specified Data Distribution '{decoder.attribute_distributions[i]}'"
                                      f" Invalid or Not Currently Implemented")
    neg_cross_entropy_Fx = torch.stack(neg_cross_entropy_Fx, dim=-1).to(logits_Fx) # (B, D)

    if permutations is not None:
        neg_cross_entropy_A = neg_cross_entropy_A.reshape(-1, permutations.shape[0]) # (B*P,) -> (B,P)
        neg_cross_entropy_A, _ = torch.max(neg_cross_entropy_A, dim=1) # (B,P,) -> (B,)

    # ELBO for adjacency matrix
    elbos = elbo_coeffs.A * neg_cross_entropy_A \
            + torch.tensor(elbo_coeffs.Fx).to(logits_Fx) @ neg_cross_entropy_Fx \
            - elbo_coeffs.beta * kld  # (B,)

    return elbos


class GraphGCNEncoder(nn.Module):

    def __init__(self, config, shared_params):
        super().__init__()
        self.config = config
        self.shared_params = shared_params

        # Create all layers except last
        self.model = self.create_model()

        # Create separate final layers for each parameter (mean and log-variance)
        # We use log-variance to unconstrain the optimisation of the positive-only variance parameters
        #TODO: change to softplus

        self.mean = nn.Linear(self.config.mlp.layer_dim[-1], self.shared_params.latent_dim)
        self.logvar = nn.Linear(self.config.mlp.layer_dim[-1], self.shared_params.latent_dim)

    def create_model(self):
        if self.config.architecture == "GIN":
            self.gcn = GIN(num_layers=self.config.gnn.num_layers, num_mlp_layers=self.config.gnn.num_mlp_layers,
                               input_dim=self.shared_params.node_attributes_dim, hidden_dim=self.config.gnn.layer_dim,
                               output_dim=self.config.mlp.layer_dim[0], final_dropout=self.config.gnn.final_dropout,
                               learn_eps=self.config.gnn.learn_eps, graph_pooling_type=self.config.gnn.graph_pooling,
                               neighbor_pooling_type=self.config.gnn.neighbor_pooling,
                               n_nodes=self.shared_params.graph_max_nodes)
        else:
            raise NotImplementedError(f"Specified GNN architecture '{self.config.architecture}'"
                                      f" Invalid or Not Currently Implemented")

        self.flatten_layer = nn.Flatten()
        self.mlp = FC_ReLU_Network([self.gcn.output_dim, *self.config.mlp.layer_dim], output_activation=nn.ReLU)
        model = [self.gcn, self.flatten_layer, self.mlp]
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
        features = self.gcn(graph, features)

        for net in self.model[1:]:
            features = net(features)

        mean = self.mean(features)
        logvar = self.logvar(features)

        return mean, logvar

    def sample_with_reparametrisation(self, mean, logvar, *, num_samples=1):
        # Reuse the implemented code
        return sample_gaussian_with_reparametrisation(mean, logvar, num_samples=num_samples)

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

class GraphMLPDecoder(nn.Module):

    def __init__(self, config, shared_params):
        super().__init__()
        self.config = config
        self.shared_params = shared_params
        self.attribute_distributions = self.config.distributions
        self.model = self.create_model(*self.shared_params.latent_dim, *self.config.layer_dim)
        if self.config.adjacency is not None:
            self.adjacency = Network((*self.config.layer_dim[-1], self.config.output_dim.adjacency),
                                     output_activation=None)
        else:
            self.adjacency = None
        if self.config.attributes:
            self.attribute_heads = []
            for i in range(len(self.config.attributes_names)):
                self.attribute_heads.append(Network((*self.config.layer_dim[-1],
                                                     *self.config.output_dim.attributes[i]),
                                                    output_activation=None))
            else:
                self.attribute_heads = None

    def create_model(self, dims: Iterable[int]):
        self.bottleneck = FC_ReLU_Network(dims[0:2], output_activation=nn.ReLU)
        self.fc_net = FC_ReLU_Network(dims[1:], output_activation=None)
        return [self.bottleneck, self.fc_net]

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

    def log_prob(self, logits: tuple, X: tuple):
        """
        Evaluates the log_probability of X given the distributions parameters

        Args:
            logits (Tuple): probabilistic graph representation (logits_A, logits_Fx)
                logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
                logits_Fx (Tensor): shape (*, B, max_nodes, D)
            X (tuple):     Batch of Graphs (A, Fx)
                A (Tensor): shape (*, B, max_nodes, max_nodes) #TODO: decide if we reduce in this function or outside
                Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            logpx (tuple):  log-probability of X (logp_A, logp_Fx)
                logp_A (Tensor): shape (*, B)
                logp_Fx (Tensor): shape (*, B, D)
        """

        A, Fx = X
        logits_A, logits_Fx = logits
        logp_A = self.log_prob_A(logits_A, A)
        logp_Fx = self.log_prob_Fx(logits_Fx, Fx)

        return logp_A, logp_Fx

    def log_prob_A(self, logits_A, A):
        """
        Evaluates the log_probability of X given the distributions parameters

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
            A (Tensor):     Batch of adjacency matrices (*, B, max_nodes, max_nodes) #TODO: decide if we reduce in this function or outside

        Returns:
            logp_A (Tensor):  log-probability of A, a batch of shape (*, B)
        """

        # A: always Bernoulli
        return evaluate_logprob_bernoulli(A, logits=logits_A)

    def log_prob_Fx(self, logits_Fx, Fx):
        """
        Evaluates the log_probability of X given the distributions parameters

        Args:
            logits_Fx (Tensor): probabilistic attribute matrix representation
            Fx (Tensor):     Batch of attribute matrices shape (*, B, max_nodes, D)

        Returns:
            logp_Fx (Tensor):  log-probability of Fx, a batch of shape (*, B, D)
                                Note: returning (*, B, D) to give option of individually weighting logprobs per attribute
        """

        # Fx #TODO: figure how to explicitly include the distributions_domains property.
        logp_Fx = []
        for i in range(len(self.attribute_distributions)):
            Fx_dim = self.config.attributes_mapping[i]
            if self.attribute_distributions[i] == "bernoulli":
                logp_Fx.append(evaluate_logprob_bernoulli(Fx[..., Fx_dim], logits=logits_Fx[..., i]))
            # Note: more efficient way to do this is to "bundle" start and goal within a single batch (i.e. B,D),
            # but requires modifying evaluate_logprob_one_hot_categorical()
            elif self.attribute_distributions[i] == "one_hot_categorical":
                logp_Fx.append(evaluate_logprob_one_hot_categorical(Fx[..., Fx_dim], logits=logits_Fx[..., i]))
            else:
                raise NotImplementedError(f"Specified Data Distribution '{self.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
        logp_Fx = torch.stack(logp_Fx, dim=-1).to(logits_Fx)
        return logp_Fx

    # Some extra methods for analysis

    def sample(self, logits: tuple, *, num_samples=1):
        """
        Samples a graph representation from the probabilistic graph tuple

        Args:
            logits (Tuple): probabilistic graph representation (logits_A, logits_Fx)
                logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
                logits_Fx (Tensor): shape (*, B, max_nodes, D)
            num_samples (int): number of samples

        Returns:
            A (Tensor):  reduced adjacency matrix, shape (num_samples, *, B, max_nodes, reduced_edges)
            Fx (Tensor): attribute matrix, shape (num_samples, *, B, max_nodes, D)
        """

        A = self.sample_A_red(logits[0], num_samples=num_samples)
        Fx = self.sample_Fx(logits[1], num_samples=num_samples)

        return A, Fx

    def sample_A_red(self, logits_A: torch.tensor, *, num_samples=1):
        """
        Samples reduced adjacency matrix from non-normalised probabilities outputted by decoder

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
            num_samples (int): number of samples

        Returns:
            A (Tensor):  reduced adjacency matrix, shape (num_samples, *, B, max_nodes, reduced_edges)
        """

        return torch.distributions.Bernoulli(logits=logits_A).sample((num_samples,))

    def sample_Fx(self, logits: torch.tensor, *, num_samples=1):
        """
        Samples attribute matrix from non-normalised probabilities outputted by decoder

        Args:
            logits (Tensor): shape (*, B, max_nodes, D)
            num_samples (int): number of samples

        Returns:
            A (Tensor):  reduced adjacency matrix, shape (num_samples, *, B, max_nodes, reduced_edges)
            Fx (Tensor): attribute matrix, shape (num_samples, *, B, max_nodes, D)
        """

        # TODO: work a way to include attributes_mapping
        # [e, s, g]
        Fx = []
        for i in range(len(self.attribute_distributions)):
            if self.attribute_distributions[i] == "bernoulli":
                empties = torch.distributions.Bernoulli(logits=logits[..., i]).sample((num_samples,))
                Fx.append(empties)
            elif self.attribute_distributions[i] == "one_hot_categorical":
                Fx.append(torch.distributions.OneHotCategorical(logits=logits[..., i]).sample((num_samples,)))
            else:
                raise NotImplementedError(f"Specified Data Distribution '{self.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
        Fx = torch.stack(Fx, dim=-1).to(logits)

        return Fx

    def param_p(self, logits: tuple):
        """
        Returns the distributions parameters

        Args:
            logits (Tuple): probabilistic graph representation (logits_A, logits_Fx)
                logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
                logits_Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            pA (Tensor): parameters of the reduced adjacency matrix distribution, shape (*, B, max_nodes, reduced_edges)
            pFx (Tensor): parameters of the feature matrix distribution, shape (*, B, max_nodes, D)
        """

        logits_A, logits_Fx = logits
        pA = self.param_pA(logits_A)
        pFx = self.param_pFx(logits_Fx)

        return pA, pFx

    def param_pA(self, logits_A):
        """
        Returns the distributions parameters

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)

        Returns:
            pA (Tensor): parameters of the reduced adjacency matrix distribution, shape (*, B, max_nodes, reduced_edges)
        """

        return torch.distributions.Bernoulli(logits=logits_A).probs

    def param_pFx(self, logits_Fx):
        """
        Returns the distributions parameters

        Args:
            logits_Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            pFx (Tensor): parameters of the feature matrix distribution, shape (*, B, max_nodes, D)
        """

        pFx = []
        for i in range(len(self.attribute_distributions)):
            if self.attribute_distributions[i] == "bernoulli":
                dist = torch.distributions.Bernoulli(logits=logits_Fx[..., i])
            elif self.attribute_distributions[i] == "one_hot_categorical":
                dist = torch.distributions.OneHotCategorical(logits=logits_Fx[..., i])
            else:
                raise NotImplementedError(f"Specified Data Distribution '{self.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
            pFx.append(dist.probs)
        pFx = torch.stack(pFx, dim=-1).to(logits_Fx)

        return pFx

    def param_m(self, logits: tuple, threshold: float = 0.5):
        """
        Returns the mode given the distribution parameters. An optional threshold parameter
        can be specified to tune cutoff point between sampling 0 or 1 (only applied to Bernoulli distributions).

        Args:
            logits (Tuple): probabilistic graph representation (logits_A, logits_Fx)
                logits_A (Tensor): reduced probabilistic adjacency matrix logits, shape (*, B, max_nodes, reduced_edges)
                logits_Fx (Tensor): probabilistic feature attributes matrix logits, shape (*, B, max_nodes, D)
            :param threshold (float): Threshold for binary transformation of Bernoulli distributions, [0,1]
                                      when threshold = 0.5, this is equivalent to taking the mode of the distribution

        Returns:
            m (Tuple): mode of distributions parameters (mA, mFx)
                mA (Tensor): mode of reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
                mFx (Tensor): mode of probabilistic feature attributes matrix, shape (*, B, max_nodes, D)
        """

        logits_A, logits_Fx = logits
        mA = self.param_mA(logits_A, threshold=threshold)
        mFx = self.param_mFx(logits_Fx, threshold=threshold)

        return mA, mFx

    def param_mA(self, logits_A, threshold: float = 0.5):
        """
        Returns the mode of the probabilistic reduced adjacency matrix given the distribution parameters.
        An optional threshold parameter can be specified to tune cutoff point between sampling 0 or 1
        (only applied to Bernoulli distributions).

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix logits, shape (*, B, max_nodes, reduced_edges)
            :param threshold (float): Threshold for binary transformation of Bernoulli distributions, [0,1]
                                      when threshold = 0.5, this is equivalent to taking the mode of the distribution

        Returns:
            mA (Tensor): mode of reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
        """

        binary_transform = BinaryTransform(threshold)
        return binary_transform(self.param_pA(logits_A))

    def param_mFx(self, logits_Fx, threshold: float = 0.5):
        """
        Returns the mode given the distribution parameters. An optional threshold parameter
        can be specified to tune cutoff point between sampling 0 or 1 (only applied to Bernoulli distributions)

        Args:
            logits_Fx (Tensor): probabilistic feature attributes matrix logits, shape (*, B, max_nodes, D)
            :param threshold (float): Threshold for binary transformation of Bernoulli distributions, [0,1]
                                      when threshold = 0.5, this is equivalent to taking the mode of the distribution

        Returns:
            mFx (Tensor): mode of probabilistic feature attributes matrix, shape (*, B, max_nodes, D)
        """

        binary_transform = BinaryTransform(threshold)
        pFx = self.param_pFx(logits_Fx)

        mFx = []
        for i in range(len(self.attribute_distributions)):
            if self.attribute_distributions[i] == "bernoulli":
                mfx = binary_transform(pFx[..., i])
            elif self.attribute_distributions[i] == "one_hot_categorical":
                mfx = pFx[..., i].argmax(axis=-1)
                mfx = F.one_hot(mfx, num_classes=pFx[..., i].shape[-1]).to(logits_Fx)
            else:
                raise NotImplementedError(f"Specified Data Distribution '{self.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
            mFx.append(mfx)

        mFx = torch.stack(mFx, dim=-1).to(logits_Fx)
        return mFx


class GraphVAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = GraphGCNEncoder(config.configuration.encoder, config.configuration.shared_parameters)
        self.decoder = GraphMLPDecoder(config.configuration.decoder, config.configuration.shared_parameters)

        if self.config.model_config.configuration.model.augmented_inputs:
            transforms = torch.tensor(self.config.configuration.model.transforms, dtype=torch.int)
            self.permutations = Batch.augment_adj(self.config.configuration.shared_parameters.graph_max_nodes,
                                                  transforms).long()
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
            return self.elbo(X)
        else:
            raise ValueError(f'gradient_type={self.hparams.gradient_type} is invalid')

    def elbo(self, X):
        return graphVAE_elbo_pathwise(X, encoder=self.encoder, decoder=self.decoder,
                                      num_samples=self.config.configuration.model.num_variational_samples,
                                      elbo_coeffs=self.config.hyperparameters.loss.elbo_coeffs,
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

