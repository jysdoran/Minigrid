import dgl
import numpy as np
import torch
import einops
import pytorch_lightning as pl
import hydra
import wandb
from torch import nn
import torch.nn.functional as F
from typing import Iterable

from models.gnn_networks import GIN
from models.networks import FC_ReLU_Network
from util.distributions import sample_gaussian_with_reparametrisation, compute_kld_with_standard_gaussian, \
    evaluate_logprob_bernoulli, evaluate_logprob_one_hot_categorical, evaluate_logprob_diagonal_gaussian
import util.transforms as tr
import util.graph_metrics as gm


def graphVAE_elbo_pathwise(X, *, encoder, decoder, num_samples, elbo_coeffs, predictor=None, labels=None, permutations=None):
    # TODO: permutations not functional at the moment, because not implemented for Fx

    mean, std = encoder(X)  # (B, H), (B, H)

    # Sample the latents using the reparametrisation trick
    Z = sample_gaussian_with_reparametrisation(
        mean, std, num_samples=num_samples)  # (M, B, H)

    # Compute KLD( q(z|x) || p(z) )
    kld = compute_kld_with_standard_gaussian(mean, std)  # (B,)

    # Evaluate the decoder network to obtain the parameters of the
    # generative model p(x|z)

    # logits = torch.randn(num_samples, *X.shape) for testing
    logits_A, logits_Fx = decoder(Z)  # (M, B, n_nodes-1, 2)

    graphs = dgl.unbatch(X)
    n_nodes = decoder.shared_params.graph_max_nodes
    reconstructed_features = tuple(decoder.attributes_mapping)
    f_dim = len(reconstructed_features) #only the reconstructed dimensions [empty, start, goal]
    A_in = torch.empty((len(graphs), n_nodes, n_nodes)).to(logits_A)
    if logits_Fx is not None:
        Fx = torch.empty((len(graphs), n_nodes, f_dim)).to(logits_Fx)
    else:
        Fx = None

    #TODO: find a way to do this efficiently on GPU, maybe convert logits to sparse tensor
    # is list comprehension plus torch.stack more efficient?
    for m in range(len(graphs)):
        A_in[m] = graphs[m].adj().to_dense()
        if Fx is not None:
            Fx[m] = graphs[m].ndata['feat'][..., reconstructed_features].to(logits_Fx)

    if permutations is not None:
        permutations = permutations.to(logits_A.device)
        A_in = [A_in[:,permutations[i]][:,:, permutations[i]] for i in range(permutations.shape[0])]
        A_in = torch.stack(A_in, dim=1).to(logits_A.device)  # B, P, N, N
        A_in = A_in.reshape(A_in.shape[0]*A_in.shape[1],*A_in.shape[2:]) #B*P, N, N
        logits_A = logits_A.repeat_interleave(permutations.shape[0], dim=1) # (M, B, n_nodes-1, 2) -> (M, B*P, n_nodes-1, 2)

    A_in = tr.Nav2DTransforms.encode_adj_to_reduced_adj(A_in) #B*P, n_nodes - 1, 2

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)

    A_in = A_in.reshape(A_in.shape[0], -1) # (B | B * P, d1, d2, ..., dn) -> (B | B*P, D=2*(n_nodes - 1))
    logits_A = logits_A.reshape(*logits_A.shape[0:2], -1) # (M, B, n_nodes-1, 2) -> (M, B, D=2*(n_nodes - 1))
    neg_cross_entropy_A = evaluate_logprob_bernoulli(A_in, logits=logits_A)  # (M, B, D)->(M,B) | (M, B*P, D)->(M,B*P)
    logits_A = logits_A.mean(dim=0)

    if permutations is not None:
        neg_cross_entropy_A = neg_cross_entropy_A.reshape(neg_cross_entropy_A.shape[0], -1, permutations.shape[0]) # (M,B*P,) -> (M,B,P)
        neg_cross_entropy_A, _ = torch.max(neg_cross_entropy_A, dim=1) # (M,B,P,) -> (M,B)

    if logits_Fx is not None:
        neg_cross_entropy_Fx = []
        for i in range(len(decoder.attribute_distributions)):
            if decoder.attribute_distributions[i] == "bernoulli":
                neg_cross_entropy_Fx.append(evaluate_logprob_bernoulli(Fx[..., i], logits=logits_Fx[..., i])) # (M,B)
            # Note: more efficient way to do this is to "bundle" start and goal within a single batch (i.e. M,B,D),
            # but requires modifying evaluate_logprob_one_hot_categorical()
            elif decoder.attribute_distributions[i] == "one_hot_categorical":
                neg_cross_entropy_Fx.append(evaluate_logprob_one_hot_categorical(Fx[..., i], logits=logits_Fx[..., i]))
            else:
                raise NotImplementedError(f"Specified Data Distribution '{decoder.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
        neg_cross_entropy_Fx = torch.stack(neg_cross_entropy_Fx, dim=-1).to(logits_Fx) # (M, B, D) [to allow for separate coeffs between start and goal]
    else:
        neg_cross_entropy_Fx = None

    # ELBO for adjacency matrix
    if logits_Fx is not None:
        #elbos_Fx = torch.einsum('i, m b i -> m b', torch.tensor(elbo_coeffs.Fx).to(logits_Fx), neg_cross_entropy_Fx)
        elbos_Fx = elbo_coeffs.Fx * neg_cross_entropy_Fx.mean(dim=-1) #(M, B, D) -> (M, B)
        logits_Fx = logits_Fx.mean(dim=0)
    else:
        elbos_Fx = 0.

    elbos = elbo_coeffs.A * neg_cross_entropy_A + elbos_Fx - elbo_coeffs.beta * kld  # (M,B,)
    unweighted_elbos = neg_cross_entropy_A + neg_cross_entropy_Fx.mean(dim=-1) - kld

    # ref: IWAE
    # - https://arxiv.org/pdf/1509.00519.pdf
    # - https://github.com/Gabriel-Macias/iwae_tutorial
    if num_samples > 1:
        elbos = torch.logsumexp(elbos, dim=0) - np.log(num_samples) # (M,B) -> (B), normalising by M to get logmeanexp()
        unweighted_elbos = torch.logsumexp(unweighted_elbos, dim=0) - np.log(num_samples) # (M,B,) -> (B,)
    else:
        elbos = elbos.mean(dim=0)
        unweighted_elbos = unweighted_elbos.mean(dim=0)

    # Add the prediction loss
    if predictor is not None:
        if predictor.target_from == "input":
            assert labels is not None, "Prediction from input requires labels."
            # y =
        else:
            reconstructed_graphs, start_nodes, goal_nodes, is_valid = \
                tr.Nav2DTransforms.encode_decoder_output_to_graph(logits_A, logits_Fx, decoder,
                                                                  correct_A=True)
            y = predictor.target_metric_fn(reconstructed_graphs, start_nodes, goal_nodes).to(Z.device)
            y = einops.repeat(y, 'b -> m b 1', m=num_samples) # (B,) -> (M,B,1)
        predictor_loss = predictor.loss(Z, y)
        elbos = elbos - elbo_coeffs.predictor * predictor_loss

        # for logging only
        y_hat = predictor(Z)
        predictor_loss_fn = predictor.loss_fn(reduction="none")
        predictor_loss_unreg = predictor_loss_fn(y_hat, y).mean(dim=0).squeeze() # (M, B, 1) -> (B,)
        y_hat = y_hat.mean(dim=0).squeeze() # (M, B, 1) -> (B,)
        y = y.mean(dim=0).squeeze()  # (M, B, 1) -> (B,)
    else:
        predictor_loss = torch.zeros(1).to(Z)
        predictor_loss_unreg = torch.zeros(elbos.shape[0]).to(Z)
        y_hat = torch.zeros(elbos.shape[0]).to(Z)
        y = torch.zeros(elbos.shape[ 0 ]).to(Z)

    outputs = {
        "loss": -elbos.mean().reshape(1),
        "elbos": elbos,
        "unweighted_elbos": unweighted_elbos,
        "neg_cross_entropy_A": neg_cross_entropy_A.mean(dim=0),
        "neg_cross_entropy_Fx": neg_cross_entropy_Fx.mean(dim=0),
        "kld": kld.mean().reshape(1),
        "predictor_loss": predictor_loss.reshape(1),
        "predictor_loss_unreg": predictor_loss_unreg,
        "y": y,
        "y_hat": y_hat,
        "logits_A": logits_A,
        "logits_Fx": logits_Fx,
        "std": std,
        "mean": mean,
        }

    #return elbos, unweighted_elbos, logits_A, logits_Fx, mean, std, y_hat, predictor_loss_unreg
    return outputs

class GraphGCNEncoder(nn.Module):

    def __init__(self, config, shared_params):
        super().__init__()
        self.config = config
        self.shared_params = shared_params
        if self.config.attributes is None or len(self.config.attributes) == 0 or self.config.attributes[0] == "":
            self.attributes = None
            self.attributes_mapping = None
        else:
            self.attributes = self.config.attributes
            self.attributes_mapping = [self.shared_params.node_attributes.index(i) for i in self.attributes]

        # Create all layers except last
        self.model = self.create_model()

        # Create separate final layers for each parameter (mean and log-variance)
        # We use log-variance to unconstrain the optimisation of the positive-only variance parameters

        self.mean = nn.Linear(self.config.mlp.bottleneck_dim, self.shared_params.latent_dim)
        self.std = FC_ReLU_Network([self.config.mlp.bottleneck_dim, self.shared_params.latent_dim],
                                   output_activation=nn.Softplus)

    def create_model(self):
        if self.config.gnn.architecture == "GIN":
            self.gcn = GIN(num_layers=self.config.gnn.num_layers, num_mlp_layers=self.config.gnn.num_mlp_layers,
                               input_dim=len(self.attributes_mapping), hidden_dim=self.config.gnn.layer_dim,
                               output_dim=self.config.mlp.hidden_dim, final_dropout=self.config.gnn.final_dropout,
                               learn_eps=self.config.gnn.learn_eps, graph_pooling_type=self.config.gnn.graph_pooling,
                               neighbor_pooling_type=self.config.gnn.neighbor_pooling,
                               n_nodes=self.shared_params.graph_max_nodes)
        else:
            raise NotImplementedError(f"Specified GNN architecture '{self.config.architecture}'"
                                      f" Invalid or Not Currently Implemented")

        self.flatten_layer = nn.Flatten()
        mlp_dims = [self.gcn.output_dim]
        mlp_dims.extend([self.config.mlp.hidden_dim] * (self.config.mlp.num_layers - 1))
        mlp_dims.extend([self.config.mlp.bottleneck_dim])
        self.mlp = FC_ReLU_Network(mlp_dims, output_activation=nn.ReLU)
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
            std (Tensor): std of the diagonal Gaussian variational distribution, shape (B, K)
        """

        graph = X
        features = graph.ndata['feat'].float()
        features = features[..., self.attributes_mapping]
        features = self.gcn(graph, features)

        for net in self.model[1:]:
            features = net(features)

        mean = self.mean(features)
        std = self.std(features)

        return mean, std

    def sample_with_reparametrisation(self, mean, std, *, num_samples=1):
        # Reuse the implemented code
        return sample_gaussian_with_reparametrisation(mean, std, num_samples=num_samples)

    def log_prob(self, mean, std, Z):
        """
        Evaluates the log_probability of Z given the parameters of the diagonal Gaussian

        Args:
            mean (Tensor):   means of the variational distributions, shape (*, B, K)
            std (Tensor): std of the diagonal Gaussian variational distribution, shape (*, B, K)
            Z (Tensor):      latent vectors, shape (*, B, K)

        Returns:
            logqz (Tensor):  log-probability of Z, a batch of shape (*, B)
        """
        # Reuse the implemented code
        return evaluate_logprob_diagonal_gaussian(Z, mean=mean, std=std)

class GraphMLPDecoder(nn.Module):

    def __init__(self, config, shared_params):

        super().__init__()
        self.config = config
        self.shared_params = shared_params
        if self.config.attributes is None or len(self.config.attributes) == 0 or self.config.attributes[0] == "":
            self.attributes = None
            self.attributes_mapping = None
        else:
            self.attributes = self.config.attributes
            self.attributes_mapping = [self.shared_params.node_attributes.index(i) for i in self.attributes]

        self.attribute_distributions = self.config.distributions

        # model creation
        hidden_dims = [self.shared_params.latent_dim, self.config.bottleneck_dim]
        if self.config.num_layers > 1:
            hidden_dims.extend([self.config.hidden_dim] * (self.config.num_layers - 1))
        # No hidden layer if the number of layers is 1
        else:
            self.config.hidden_dim = self.config.bottleneck_dim
        self.model = self.create_model(hidden_dims)
        if self.config.adjacency is not None:
            self.adjacency = nn.Linear(self.config.hidden_dim, self.config.output_dim.adjacency)
        else:
            self.adjacency = None


        if self.attributes is not None:
            self.attribute_heads = nn.ModuleList()
            for i in range(len(self.attributes)):
                self.attribute_heads.append(nn.Linear(self.config.hidden_dim, self.config.output_dim.attributes))
        else:
            self.attribute_heads = None

    def create_model(self, dims: Iterable[int]):
        self.bottleneck = FC_ReLU_Network(dims[0:2], output_activation=nn.ReLU)
        if len(dims) > 2:
            self.fc_net = FC_ReLU_Network(dims[1:], output_activation=nn.ReLU)
            return [self.bottleneck, self.fc_net]
        else:
            return [self.bottleneck]

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
            f_out = torch.stack(f_out, dim=-1)
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
                A (Tensor): shape (*, B, max_nodes, reduced_edges)
                Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            logpx (tuple):  log-probability of X (logp_A, logp_Fx)
                logp_A (Tensor): shape (*, B)
                logp_Fx (Tensor): shape (*, B, D)
        """

        A, Fx = X
        logits_A, logits_Fx = logits
        logp_A = self.log_prob_A(logits_A, A)
        if logits_Fx is not None:
            logp_Fx = self.log_prob_Fx(logits_Fx, Fx)
        else:
            logp_Fx = None

        return logp_A, logp_Fx

    def log_prob_A(self, logits_A, A):
        """
        Evaluates the log_probability of X given the distributions parameters

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
            A (Tensor):     Batch of adjacency matrices (*, B, max_nodes, reduced_edges)

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
            Fx_dim = self.attributes_mapping[i]
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
        if logits[1] is not None:
            Fx = self.sample_Fx(logits[1], num_samples=num_samples)
        else:
            Fx = None

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
        if logits_Fx is not None:
            pFx = self.param_pFx(logits_Fx)
        else:
            pFx = None

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

    def entropy(self, logits:tuple):
        """
        Returns the distributions entropy

        Args:
            logits (Tuple): probabilistic graph representation (logits_A, logits_Fx)
                logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)
                logits_Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            H_A (Tensor): Entropy of the distribution, shape (1)
            H_Fx (Tensor): Entropy of the distribution, shape (D)
        """

        logits_A, logits_Fx = logits
        H_A = self.entropy_A(logits_A)
        if logits_Fx is not None:
            H_Fx = self.entropy_Fx(logits_Fx)
        else:
            H_Fx = None

        return H_A, H_Fx

    def entropy_A(self, logits_A):
        """
        Returns the distributions parameters

        Args:
            logits_A (Tensor): reduced probabilistic adjacency matrix, shape (*, B, max_nodes, reduced_edges)

        Returns:
            H_A (Tensor): Entropy of the distribution, shape (1)
        """

        return torch.distributions.Bernoulli(logits=logits_A).entropy().mean()

    def entropy_Fx(self, logits_Fx):
        """
        Returns the distributions parameters

        Args:
            logits_Fx (Tensor): shape (*, B, max_nodes, D)

        Returns:
            H_Fx (Tensor): Entropy of the distribution, shape (D)
        """

        H_Fx = []
        for i in range(len(self.attribute_distributions)):
            if self.attribute_distributions[i] == "bernoulli":
                dist = torch.distributions.Bernoulli(logits=logits_Fx[..., i])
            elif self.attribute_distributions[i] == "one_hot_categorical":
                dist = torch.distributions.OneHotCategorical(logits=logits_Fx[..., i])
            else:
                raise NotImplementedError(f"Specified Data Distribution '{self.attribute_distributions[i]}'"
                                          f" Invalid or Not Currently Implemented")
            H_Fx.append(dist.entropy().mean())
        H_Fx = torch.stack(H_Fx, dim=-1).to(logits_Fx)

        return H_Fx

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
        if logits_Fx is not None:
            mFx = self.param_mFx(logits_Fx, threshold=threshold)
        else:
            mFx = None

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

        binary_transform = tr.BinaryTransform(threshold)
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

        binary_transform = tr.BinaryTransform(threshold)
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

class Predictor(nn.Module):

    def __init__(self, config, shared_params) :

        super().__init__()
        self.config = config
        self.shared_params = shared_params

        # model creation
        dims = [self.shared_params.latent_dim]
        if self.config.hidden_dim is not None and self.config.num_layers > 1:
            dims.extend([self.config.hidden_dim] * (self.config.num_layers - 1))
        dims.append(1)
        self.model = self.create_model(dims)
        if self.config.target_metric == "resistance_distance":
            self.target_metric_fn = self.get_resistance_distance
            self.loss_fn = nn.MSELoss
            self._compute_loss_fn = self.loss_fn()
        else:
            raise NotImplementedError(f"Specified Target Metric '{self.config.target_metric}'"
                                      f" Invalid or Not Currently Implemented.")
        if self.config.target_from in ["input", "output"]:
            self.target_from = self.config.target_from
        else:
            raise NotImplementedError(f"Specified Target Acquisition '{self.config.target_from}'"
                                      f" Invalid or Not Currently Implemented.")

    def create_model(self, dims: Iterable[ int ]) :
        return FC_ReLU_Network(dims, output_activation=None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

    def loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.forward(z)
        loss = self._compute_loss_fn(y_hat, y)
        if self.config.alpha_reg > 0:
            l2_norm = sum(torch.linalg.norm(p, 2) for p in self.model.parameters())
            loss += self.config.alpha_reg * l2_norm
        return loss

    def get_resistance_distance(self, graphs, start_nodes, goal_nodes):
        if isinstance(graphs, dgl.DGLGraph):
            graphs = dgl.unbatch(graphs)

        metrics = []
        for b in range(0, len(graphs)):
            start = start_nodes[b].item()
            goal = goal_nodes[b].item()
            graph, valid, connected = gm.prepare_graph(graphs[b], start, goal)
            if valid and connected:
                metric = gm.resistance_distance(graph, start, goal)
                if metric == np.NaN:
                    metric = 0.
            else:
                metric = 0.
            metrics.append(metric)

        metrics = self.transform_target(torch.tensor(metrics).to(graphs[0].device, dtype=torch.float32))

        return metrics

    def transform_target(self, target, eps=1e-5):
        if self.config.target_transform == "log":
            return torch.log(target + eps)
        elif self.config.target_transform == "identity":
            return target
        else:
            raise NotImplementedError(f"Specified Target Transform '{self.config.target_transform}'"
                                      f" Invalid or Not Currently Implemented.")


class GraphVAE(nn.Module):
    """
    A wrapper for the VAE model
    """

    def __init__(self, configuration, hyperparameters, **kwargs):
        super().__init__()
        self.configuration = configuration
        self.hyperparameters = hyperparameters

        self.encoder = GraphGCNEncoder(self.configuration.encoder, self.configuration.shared_parameters)
        self.decoder = GraphMLPDecoder(self.configuration.decoder, self.configuration.shared_parameters)

        if self.configuration.model.augmented_inputs:
            transforms = torch.tensor(self.configuration.model.transforms, dtype=torch.int)
            self.permutations = tr.Nav2DTransforms.augment_adj(self.configuration.shared_parameters.graph_max_nodes,
                                                  transforms).long()
        else: self.permutations = None
        self.device = torch.device("cuda" if configuration.model.cuda else "cpu")
        self.to(self.device)


    def forward(self, X):
        """
        Computes the variational ELBO

        Args:
            X (Tensor):  data, a batch of shape (B, K)

        Returns:
            elbos (Tensor): per data-point elbos, shape (B, D)
        """
        if self.configuration.model.gradient_type == 'pathwise':
            return self.elbo(X)
        else:
            raise ValueError(f'gradient_type={self.hparams.gradient_type} is invalid')

    def all_model_outputs_pathwise(self, X, num_samples: int = None):
        if num_samples is None: num_samples = self.configuration.model.num_variational_samples
        outputs = \
            graphVAE_elbo_pathwise(X, encoder=self.encoder, decoder=self.decoder,
                                 num_samples=num_samples,
                                 elbo_coeffs=self.hyperparameters.loss.elbo_coeffs,
                                 permutations=self.permutations)
        return outputs

    def elbo(self, X):
        outputs = self.all_model_outputs_pathwise(X)
        return outputs["elbos"]

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


class LightningGraphVAE(pl.LightningModule):

    def __init__(self, config_model, config_optim, hparams_model, config_logging, **kwargs):
        super(LightningGraphVAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = GraphGCNEncoder(self.hparams.config_model.encoder, self.hparams.config_model.shared_parameters)
        self.decoder = GraphMLPDecoder(self.hparams.config_model.decoder, self.hparams.config_model.shared_parameters)
        if self.hparams.config_model.predictor.enable:
            self.predictor = Predictor(self.hparams.config_model.predictor, self.hparams.config_model.shared_parameters)
        else:
            self.predictor = None

        if self.hparams.config_model.model.augmented_inputs:
            transforms = torch.tensor(self.hparams.config_model.model.transforms, dtype=torch.int)
            self.permutations = tr.Nav2DTransforms.augment_adj(self.hparams.config_model.shared_parameters.graph_max_nodes,
                                                  transforms).long()
        else: self.permutations = None

        device = torch.device("cuda" if self.hparams.config_model.model.accelerator == "gpu" else "cpu")
        self.to(device)

    def forward(self, X):
        return self.elbo(X)

    def all_model_outputs_pathwise(self, X, num_samples: int = None):
        if num_samples is None: num_samples = self.hparams.config_model.model.num_variational_samples
        outputs = \
        graphVAE_elbo_pathwise(X, encoder=self.encoder, decoder=self.decoder, predictor=self.predictor,
                                 num_samples=num_samples,
                                 elbo_coeffs=self.hparams.hparams_model.loss.elbo_coeffs,
                                 permutations=self.permutations)
        return outputs

    def elbo(self, X):
        outputs = self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs["elbos"]

    def loss(self, X):
        outputs = self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs["loss"]

    def training_step(self, batch, batch_idx):
        X, labels = batch
        loss = self.loss(X)

        self.log('loss/train', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx, **kwargs):
        X, labels = batch
        outputs = \
            self.all_model_outputs_pathwise(X, num_samples=self.hparams.config_model.model.num_variational_samples)
        return outputs

    def predict_step(self, batch, batch_idx, **kwargs):
        dataloader_idx = 0
        return self.validation_step(batch, batch_idx, dataloader_idx, **kwargs)

    def test_step(self, batch, batch_idx, **kwargs):
        dataloader_idx = 0
        return self.validation_step(batch, batch_idx, dataloader_idx, **kwargs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.config_optim, params=self.parameters())
        return optimizer

    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams)