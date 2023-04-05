import logging
import copy
from collections import defaultdict

import dgl
import numpy as np
import torch
import einops
import pytorch_lightning as pl
import hydra
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Union, List, Dict, Tuple
import networkx as nx

from . import layers as L
from .gnn_networks import GIN
from .networks import FC_ReLU_Network
from ..util.distributions import sample_gaussian_with_reparametrisation, compute_kld_with_standard_gaussian, \
    evaluate_logprob_bernoulli, evaluate_logprob_one_hot_categorical, evaluate_logprob_diagonal_gaussian
from ..util import transforms as tr
from ..util import graph_metrics as gm
from ..util import util

logger = logging.getLogger(__name__)

def graphVAE_elbo_pathwise(X, *, encoder, decoder, num_samples, elbo_coeffs, output_keys, predictor=None, labels=None,
                           permutations=None):
    """
    Calculates the pathwise ELBO for a graph VAE model, given a set of input graphs X, a trained encoder and decoder
    networks, the number of samples to use in the Monte Carlo estimate of the ELBO, and the coefficients to use for
    different terms in the ELBO.

    Parameters
    ----------
    X : DGLGraph or list of DGLGraph
        Input graphs. If a list, must have length B, where B is the batch size.
    encoder : nn.Module
        Encoder network for the graph VAE.
    decoder : nn.Module
        Decoder network for the graph VAE.
    num_samples : int
        Number of samples to use in the Monte Carlo estimate of the ELBO.
    elbo_coeffs : dict
        Dictionary of coefficients to use for different terms in the ELBO. Must contain keys 'KLD' and 'Fx', where
        'KLD' maps to a float and 'Fx' maps to either a float or a list of floats with length equal to the number of
        feature heads in the decoder. If a float is provided for 'Fx', the same coefficient will be used for all
        feature heads.
    output_keys : list of str
        List of keys to use for the output dictionary. Must contain at least 'KLD' and 'Fx'.
    predictor : nn.Module, optional
        Predictor network for semi-supervised learning, by default None.
    labels : torch.Tensor, optional
        Labels for semi-supervised learning, by default None.
    permutations : torch.Tensor, optional
        Permutations to use for the input graphs. If not None, must be a tensor of size (P, N), where P is the number of
        permutations and N is the maximum number of nodes in the graphs. Each row of the tensor should contain a
        permutation of the node indices.

    Returns
    -------
    dict
        Dictionary containing the specified outputs.

    Raises
    ------
    ValueError
        If an invalid key is provided in output_keys.
    """

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
    logits = decoder(Z)  # (M, B, n_nodes-1, 2)
    # if decoder.adjacency is None and decoder.attributes is not None:
    #     logits_Fx = logits
    #     logits_A = None
    # elif decoder.adjacency is not None and decoder.attributes is None:
    #     logits_Fx = None
    #     logits_A = logits
    # elif decoder.adjacency is not None and decoder.attributes is not None:
    #     logits_A, logits_Fx = logits
    # else:
    #     raise ValueError('Decoder must have at least one of adjacency or attributes head')

    # Feature head(s) computations
    # if logits_Fx is not None:
    Fx = decoder.get_node_features(X, node_attributes=decoder.attributes, as_tensor=False)
    # mask Fx if needed
    if decoder.config.attribute_masking == "always":
        mask = decoder.compute_attribute_mask_from_Fx(Fx=Fx)
        logits = decoder.masked_logits(logits, mask)
    else:
        mask = None

    # Compute ~E_{q(z|x)}[ p(x | z) ]
    # Important: the samples are "propagated" all the way to the decoder output,
    # indeed we are interested in the mean of p(X|z)
    Fx_elbo = decoder.get_node_features(X, node_attributes=decoder.attributes, as_tensor=False, for_elbo=True)
    neg_cross_entropy_Fx = decoder.log_prob(logits, Fx_elbo) # (F keys: M, B,)
    neg_cross_entropy_Fx = torch.stack(list(neg_cross_entropy_Fx.values()), dim=-1).to(Z)  # (M, B, F) [to allow for separate coeffs between start and goal]

    # neg_cross_entropy_Fx = []
    # for i in range(len(decoder.output_distributions)):
    #     if decoder.output_distributions[i] == "bernoulli":
    #         neg_cross_entropy_Fx.append(evaluate_logprob_bernoulli(Fx[..., i], logits=logits_Fx[..., i])) # (M,B)
    #     # Note: more efficient way to do this is to "bundle" start and goal within a single batch (i.e. M,B,D),
    #     # but requires modifying evaluate_logprob_one_hot_categorical()
    #     elif decoder.output_distributions[i] == "one_hot_categorical":
    #         neg_cross_entropy_Fx.append(evaluate_logprob_one_hot_categorical(Fx[..., i], logits=logits_Fx[..., i]))
    #     else:
    #         raise NotImplementedError(f"Specified Data Distribution '{decoder.output_distributions[i]}'"
    #                                   f" Invalid or Not Currently Implemented")
    # neg_cross_entropy_Fx = torch.stack(neg_cross_entropy_Fx, dim=-1).to(logits_Fx) # (M, B, D) [to allow for separate coeffs between start and goal]

    if not hasattr(elbo_coeffs.Fx, '__iter__'):
        elbos_Fx = elbo_coeffs.Fx * neg_cross_entropy_Fx.mean(dim=-1)  # (M, B, F) -> (M, B)
    else:
        coeffs_Fx = list(elbo_coeffs.Fx.values())
        elbos_Fx = torch.einsum('i, m b i -> m b', torch.tensor(coeffs_Fx).to(Z), neg_cross_entropy_Fx)

    # else:
    #     Fx = None
    #     neg_cross_entropy_Fx = 0.
    #     elbos_Fx = 0

    # Adjacency head computations
    if 'adjacency' in logits.keys():
        # Get A
        # TODO: find a way to do this efficiently on GPU, maybe convert logits to sparse tensor
        # is list comprehension plus torch.stack more efficient?
        # should we try to not unbatch the graph and then reshape the big adjacency matrix?
        graphs = dgl.unbatch(X)
        n_nodes = decoder.shared_params.graph_max_nodes
        A_in = torch.empty((len(graphs), n_nodes, n_nodes)).to(logits['adjacency'])
        for m in range(len(graphs)):
            A_in[m] = graphs[m].adj().to_dense()

        # Note: permutations implemented for layout only
        if permutations is not None:
            permutations = permutations.to(logits['adjacency'].device)
            A_in = [A_in[:, permutations[i]][:, :, permutations[i]] for i in range(permutations.shape[0])]
            A_in = torch.stack(A_in, dim=1).to(logits['adjacency'].device)  # B, P, N, N
            A_in = A_in.reshape(A_in.shape[0] * A_in.shape[1], *A_in.shape[2:])  # B*P, N, N
            logits['adjacency'] = logits['adjacency'].repeat_interleave(permutations.shape[0],
                                                  dim=1)  # (M, B, n_nodes-1, 2) -> (M, B*P, n_nodes-1, 2)

        if decoder.shared_params.data_encoding == "minimal":
            A_in = tr.Nav2DTransforms.encode_adj_to_reduced_adj(A_in)  # B*P, n_nodes - 1, 2

        # Compute ~E_{q(z|x)}[ p(x | z) ]
        # Important: the samples are "propagated" all the way to the decoder output,
        # indeed we are interested in the mean of p(X|z)
        A_in = A_in.reshape(A_in.shape[0], -1)  # (B | B * P, d1, d2, ..., dn) -> (B | B*P, D=2*(n_nodes - 1))
        logits['adjacency'] = logits['adjacency'].reshape(*logits['adjacency'].shape[0:2], -1)  # (M, B, n_nodes-1, 2) -> (M, B, D=2*(n_nodes - 1))
        neg_cross_entropy_A = evaluate_logprob_bernoulli(A_in,
                                                         logits=logits['adjacency'])  # (M, B, D)->(M,B) | (M, B*P, D)->(M,B*P)

        if permutations is not None:
            neg_cross_entropy_A = neg_cross_entropy_A.reshape(neg_cross_entropy_A.shape[0], -1,
                                                              permutations.shape[0])  # (M,B*P,) -> (M,B,P)
            neg_cross_entropy_A, _ = torch.max(neg_cross_entropy_A, dim=1)  # (M,B,P,) -> (M,B)

        elbos_A = elbo_coeffs.A * neg_cross_entropy_A
    else:
        A_in = None
        neg_cross_entropy_A = 0.
        elbos_A = 0.

    for head in logits.keys():
        logits[head] = logits[head].mean(dim=0)  # (B, N)

    elbos = elbos_A + elbos_Fx - elbo_coeffs.beta * kld  # (M,B,)
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
                tr.Nav2DTransforms.encode_decoder_output_to_graph(logits, decoder,
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

    output_dict = {}
    for key in output_keys:
        if key == "loss":
            output_dict[key] = -elbos.mean().reshape(1)
        elif key == "elbos":
            output_dict[key] = elbos
        elif key == "unweighted_elbos":
            output_dict[key] = unweighted_elbos
        elif key == "neg_cross_entropy_A":
            output_dict[key] = neg_cross_entropy_A.mean(dim=0)
        elif key == "neg_cross_entropy_Fx":
            output_dict[key] = neg_cross_entropy_Fx.mean(dim=0)
        elif key == "kld":
            output_dict[key] = kld.mean().reshape(1)
        elif key == "predictor_loss":
            output_dict[key] = predictor_loss.reshape(1)
        elif key == "predictor_loss_unreg":
            output_dict[key] = predictor_loss_unreg
        elif key == "y":
            output_dict[key] = y
        elif key == "y_hat":
            output_dict[key] = y_hat
        elif key == "logits_heads":
            output_dict[key] = logits
        elif key == "logits_A":
            logits_A = logits['adjacency']
            output_dict[key] = logits_A
        elif key == "logits_Fx":
            logits_Fx = decoder.to_graph_features(logits=logits, probabilistic=True)
            # logits_Fx = torch.stack([logits[feat] for feat in decoder.output_distributions], dim=-1)
            output_dict[key] = logits_Fx
        elif key == "std":
            output_dict[key] = std
        elif key == "mean":
            output_dict[key] = mean
        else:
            raise ValueError(f"Unknown key {key}")

    return output_dict


class GraphGCNEncoder(nn.Module):

    def __init__(self, config, shared_params):
        super().__init__()
        self.config = config
        self.shared_params = shared_params
        if self.config.attributes is None or len(self.config.attributes) == 0 or self.config.attributes[0] == "":
            self.attributes = None
            logger.warning(f"No attributes specified for {self.__class__}.")
        else:
            self.attributes = self.config.attributes

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
                               input_dim=len(self.attributes), hidden_dim=self.config.gnn.layer_dim,
                               output_dim=self.config.mlp.hidden_dim, final_dropout=self.shared_params.dropout,
                               learn_eps=self.config.gnn.learn_eps, graph_pooling_type=self.config.gnn.graph_pooling,
                               neighbor_pooling_type=self.config.gnn.neighbor_pooling,
                               n_nodes=self.shared_params.graph_max_nodes,
                               enable_batch_norm=self.shared_params.use_batch_norm)
        else:
            raise NotImplementedError(f"Specified GNN architecture '{self.config.architecture}'"
                                      f" Invalid or Not Currently Implemented")

        self.flatten_layer = nn.Flatten()
        mlp_dims = [self.gcn.output_dim]
        mlp_dims.extend([self.config.mlp.hidden_dim] * (self.config.mlp.num_layers - 1))
        mlp_dims.extend([self.config.mlp.bottleneck_dim])
        self.mlp = FC_ReLU_Network(mlp_dims,
                                   output_activation=nn.ReLU,
                                   dropout=self.shared_params.dropout,
                                   batch_norm=self.shared_params.use_batch_norm,
                                   batch_norm_output_layer=self.config.mlp.batch_norm_output_layer,
                                   dropout_output_layer=self.config.mlp.dropout_output_layer)
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
        features = self.get_node_features(graph, node_attributes=self.attributes, reshape=False, as_tensor=True)
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

    def get_node_features(self, graph: Union[dgl.DGLGraph, List[dgl.DGLGraph]], node_attributes:List[str]=None,
                          reshape:bool=True, as_tensor=True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if node_attributes is None:
            node_attributes = self.attributes

        features, _ = util.get_node_features(graph, node_attributes=node_attributes, device=None,
                                                           reshape=reshape)
        if as_tensor:
            return features.float()
        else:
            features_dict = {}
            for i, key in enumerate(node_attributes):
                features_dict[key] = features[..., i].float()
            return features_dict


class GraphMLPDecoder(nn.Module):

    def __init__(self, config, shared_params):

        super().__init__()
        self.config = config
        self.shared_params = shared_params
        if self.config.attributes is None or len(self.config.attributes) == 0 or self.config.attributes[0] == "":
            self.attributes = None
            logger.warning(f"No attributes specified for {self.__class__}.")
        else:
            self.attributes = self.config.attributes

        self.output_distributions = self.config.distributions

        # model creation
        hidden_dims = [self.shared_params.latent_dim, self.config.bottleneck_dim]
        if self.config.num_layers > 1:
            hidden_dims.extend([self.config.hidden_dim] * (self.config.num_layers - 1))
        # No hidden layer if the number of layers is 1
        else:
            self.config.hidden_dim = self.config.bottleneck_dim
        self.model = self.create_model(hidden_dims)


        # create heads
        self.heads = nn.ModuleDict()
        if self.config.adjacency is not None:
            self.heads["adjacency"] = nn.Linear(self.config.hidden_dim, self.config.output_dim.adjacency)
        if self.attributes is not None:
            for head in self.output_distributions:
                output_dim = self.shared_params.graph_max_nodes * len(self.output_distributions[head].attributes)
                self.heads[head] = nn.ModuleList([
                    nn.Linear(self.config.hidden_dim, output_dim),
                    L.Reshape(-1, self.shared_params.graph_max_nodes, len(self.output_distributions[head].attributes))])

    def create_model(self, dims: Iterable[int]):
        self.bottleneck = FC_ReLU_Network(dims[0:2],
                                          output_activation=nn.ReLU,
                                          dropout=self.shared_params.dropout,
                                          batch_norm=self.shared_params.use_batch_norm,
                                          batch_norm_output_layer=self.shared_params.use_batch_norm,
                                          dropout_output_layer=bool(self.shared_params.dropout))
        if len(dims) > 2:
            self.fc_net = FC_ReLU_Network(dims[1:],
                                          output_activation=nn.ReLU,
                                          dropout=self.shared_params.dropout,
                                          batch_norm=self.shared_params.use_batch_norm,
                                          dropout_output_layer=self.config.dropout_output_layer,
                                          batch_norm_output_layer=self.config.batch_norm_output_layer)
            return [self.bottleneck, self.fc_net]
        else:
            return [self.bottleneck]

    def forward(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the parameters of the generative distribution p(x | z)

        Args:
            Z (Tensor):  latent vectors, a batch of shape (M, B, K)

        Returns:
            out (Dict[str, torch.Tensor]): logits of distributions parametrised by self.output_distributions
        """

        logits = Z.reshape(-1, Z.shape[-1])
        if logits.ndim != Z.ndim:
            reshape = True
        else:
            reshape = False

        for net in self.model:
            logits = net(logits)
            
        out = {}
        for head in self.heads:
            out[head] = logits.clone()
            for net in self.heads[head]:
                out[head] = net(out[head])
            # out[head] = self.heads[head](logits)
            if reshape:
                out[head] = out[head].reshape(*Z.shape[:-1], *out[head].shape[1:])
        return out

    def log_prob(self, logits: Dict[str, torch.Tensor], X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the log probability of the generative distribution p(x | z) given the output logits

        Args:
            logits (Dict[str, torch.Tensor]): output logits of distributions parametrised by self.output_distributions
            X (Dict[str, torch.Tensor]): input data dictionary, with keys corresponding to output distributions

        Returns:
            logp (Dict[str, torch.Tensor]): log probability dictionary of each output distribution
        Raises:
            AssertionError: If Bernoulli distribution is not over "node" domain
            NotImplementedError: If distribution family or domain is not implemented
        """

        logp = {}
        for head in logits:
            if self.output_distributions[head].family == "bernoulli":
                assert self.output_distributions[head].domain == "node", \
                    "Bernoulli distribution must be over node domain."
                logp[head] = evaluate_logprob_bernoulli(X[head].to(logits[head]), logits=logits[head])
            elif self.output_distributions[head].family == "one_hot_categorical":
                if self.output_distributions[head].domain == "nodeset":
                    feats = self.output_distributions[head].attributes
                    assert len(feats) == 1, "One-hot categorical distribution over nodeset domain must be univariate."
                    x = X[feats[0]]
                    logp[head] = evaluate_logprob_one_hot_categorical(x.to(logits[head]),
                                                                      logits=logits[head].squeeze(-1))
                elif self.output_distributions[head].domain == "node":
                    feats = self.output_distributions[head].attributes
                    x = torch.stack([X[f] for f in feats], dim=-1)
                    logp[head] = evaluate_logprob_one_hot_categorical(x.to(logits[head]),logits=logits[head])
                    # TODO: by doing this we lose ability to monitor the cross-entropy over the different nodes in layout
                    logp[head] = logp[head].mean(dim=-1)
                else:
                    raise NotImplementedError(f"Domain {self.output_distributions[head].domain} not implemented.")
            else:
                raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")
        return logp

    def sample(self, logits: Dict[str, torch.Tensor], *, num_samples=1) -> Dict[str, torch.Tensor]:
        """
        Sample from the output distribution of the generative model p(x | z)

        Args:
            logits (Dict[str, torch.Tensor]): logits of distributions parametrised by self.output_distributions
            num_samples (int): the number of samples to draw from each distribution (default: 1)

        Returns:
            samples (Dict[str, torch.Tensor]): dictionary of samples, where each sample has the same shape as the corresponding logits input.
        """

        samples = {}
        for head in logits:
            if head == "adjacency":
                samples[head] = self.sample_A_red(logits[head], num_samples=num_samples)
            else:
                if self.output_distributions[head].family == "bernoulli":
                    assert self.output_distributions[head].domain == "node", \
                        "Bernoulli distribution must be over node domain."
                    samples[head] = torch.distributions.Bernoulli(logits=logits[head]).sample((num_samples,))
                elif self.output_distributions[head].family == "one_hot_categorical":
                    if self.output_distributions[head].domain == "nodeset":
                        ll = logits[head].squeeze(-1)
                        samples[head] = torch.distributions.OneHotCategorical(logits=ll).sample((num_samples,)).unsqueeze(-1)
                    elif self.output_distributions[head].domain == "node":
                        ll = logits[head]
                        samples[head] = torch.distributions.OneHotCategorical(logits=ll).sample((num_samples,))
                    else:
                        raise NotImplementedError(f"Domain {self.output_distributions[head].domain} "
                                                  f"not implemented for distribution "
                                                  f"{self.output_distributions[head].family}.")

                else:
                    raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")
        return samples

    def param_p(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the probability parameters (probs) for each output head of the decoder, given the input logits.

        Args:
            logits (Dict[str, torch.Tensor]): A dictionary where the keys are the output head names and the values are
                                               the logits tensors.
            masked (bool, optional): Whether to mask the logits or not. Defaults to True.
            mask (torch.Tensor, optional): A binary mask tensor to apply to the logits. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where the keys are the output head names and the values are the
                                     corresponding probability parameters (probs) tensors.
                - param_p["adjacency"]: A tensor of shape (*, N, N) where * are any additional dimensions in the logits
                and N is the number of nodes in the graph.
                - For distribution heads, the shape is (*, N, C) where C is the number of classes in the distribution.

        Raises:
            RuntimeError: If `config.attribute_masking` is set to None and `mask` is not provided.
            ValueError: If `mask` is provided but `masked` is set to False.
            NotImplementedError: If the output distribution family of an output head is not implemented in the function.

        Note:
            This function assumes that an instance of the class containing it has already been created with a `config`
            attribute. The `config` attribute is used to determine whether to use auto-masking or not. If `config.attribute_masking`
            is set to None, this function requires a mask to be provided explicitly through the `mask` argument.
        """

        param_p = {}
        for head in logits:
            if self.output_distributions[head].family == "bernoulli":
                assert self.output_distributions[head].domain == "node", \
                    "Bernoulli distribution must be over node domain."
                param_p[head] = torch.distributions.Bernoulli(logits[head]).probs
            elif self.output_distributions[head].family == "one_hot_categorical":
                if self.output_distributions[head].domain == "nodeset":
                    param_p[head] = torch.distributions.OneHotCategorical(logits=logits[head].squeeze(-1)).probs.unsqueeze(-1)
                else:
                    param_p[head] = torch.distributions.OneHotCategorical(logits=logits[head]).probs
            else:
                raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")

        return param_p

    def param_m(self, logits: Dict[str, torch.Tensor] = None, probs: Dict[str, torch.Tensor] = None,
                thresholds: Dict[str, float]=None) \
            -> Dict[str, torch.Tensor]:
        """
        Transforms probabilities or logits to the corresponding parameters for the distribution family of each head.

        Args:
            logits (Dict[str, torch.Tensor]): A dictionary containing the logits for each head. Either `logits` or `probs` must be provided.
            probs (Dict[str, torch.Tensor]): A dictionary containing the probabilities for each head. Either `logits` or `probs` must be provided.
            thresholds (Dict[str, float]): A dictionary containing the threshold values for each head. If not provided, it is set to 0.5 for each head.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the transformed parameters for each head. The keys are the heads and the values are the transformed parameters.

        Raises:
            ValueError: If neither `logits` nor `probs` is provided, or if both `logits` and `probs` are provided.
            NotImplementedError: If the distribution family of a head is not implemented.
        """

        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided")
        elif probs is not None and logits is not None:
            raise ValueError("Only one of probs or logits can be provided")
        elif probs is None and logits is not None:
            probs = self.param_p(logits)

        if thresholds is None:
            thresholds = dict.fromkeys(probs.keys(), 0.5) #only for bernoulli

        param_m = {}
        for head in probs:
            if self.output_distributions[head].family == "bernoulli":
                assert self.output_distributions[head].domain == "node", \
                    "Bernoulli distribution must be over node domain."
                binary_transform = tr.BinaryTransform(thresholds[head])
                param_m[head] = binary_transform(probs[head])
            elif self.output_distributions[head].family == "one_hot_categorical":
                if self.output_distributions[head].domain == "nodeset":
                    pp = probs[head].squeeze(-1)
                elif self.output_distributions[head].domain == "node":
                    pp = probs[head]
                else:
                    raise NotImplementedError(f"Domain {self.output_distributions[head].domain} not implemented.")
                mode = pp.argmax(dim=-1)
                param_m[head] = F.one_hot(mode, num_classes=pp.shape[-1]).to(pp)
                param_m[head] = torch.reshape(param_m[head], probs[head].shape)
            else:
                raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")
        return param_m

    def entropy(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This function calculates the entropy of the distributions represented by the given logits. The logits argument should be a dictionary with keys corresponding to the different heads of the output distributions, and values corresponding to the logits of each head. The output of the function is also a dictionary, with keys corresponding to the different heads of the output distributions, and values corresponding to the entropy of each head.

        Args:
        logits (Dict[str, torch.Tensor]): A dictionary with keys corresponding to the different heads of the output
        distributions, and values corresponding to the logits of each head.

        Returns:
        Dict[str, torch.Tensor]: A dictionary with keys corresponding to the different heads of the output
        distributions, and values corresponding to the entropy of each head.

        The shape of each value will depend on the distribution family and domain of the corresponding head.
            - If the distribution family is bernoulli, the shape will be (*, max_nodes).
            - If the distribution family is one_hot_categorical and the domain is nodeset, the shape will be (*, max_nodes, 1).
            - If the distribution family is one_hot_categorical and the domain is node, the shape will be (*,).

        Where * represents the first dimension(s) of the input tensor that are reserved for parallelization
        (e.g. batchsize B, num of variational samples M).

        If the distribution family is not implemented, a NotImplementedError will be raised.
        """
        entropy = {}
        for head in logits:
            if self.output_distributions[head].family == "bernoulli":
                assert self.output_distributions[head].domain == "node", \
                    "Bernoulli distribution must be over node domain."
                entropy[head] = torch.distributions.Bernoulli(logits=logits[head]).squeeze(-1).entropy()
            elif self.output_distributions[head].family == "one_hot_categorical":
                if self.output_distributions[head].domain == "nodeset":
                    lo = logits[head].squeeze(-1)
                elif self.output_distributions[head].domain == "node":
                    lo = logits[head]
                else:
                    raise NotImplementedError(f"Domain {self.output_distributions[head].domain} not implemented.")
                entropy[head] = torch.distributions.OneHotCategorical(logits=lo).entropy()
            else:
                raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")
        return entropy

    def compute_attribute_mask_from_Fx(self, Fx:Dict[str, torch.Tensor]=None) \
            -> Dict[str, torch.Tensor]:

        """
        Computes the attribute mask for each head in the output distribution.

        The attribute mask is a boolean tensor indicating which logits/probs of the output distribution should be kept
        during decoding.

        Args:
            Fx (Dict[str, torch.Tensor], optional): Dictionary of input features for each head in the output
                distribution. The shape of each tensor should be (*, B, max_nodes, D), where * is any number of
                additional dimensions, B is the batch size, max_nodes is the maximum number of nodes in the graph,
                and D is the dimensionality of the input features.

        Returns:
            attribute_mask (Dict[str, torch.Tensor]): Dictionary of attribute masks for each head in the output
                distribution. The shape of each tensor is (*, B, max_nodes, d), where * is any number of additional
                dimensions, B is the batch size, max_nodes is the maximum number of nodes in the graph, and d is the
                dimensionality of the output distribution.
        """

        probs = {}
        for head in self.output_distributions.keys():
            probs[head] = torch.stack([Fx[attr] for attr in self.output_distributions[head].attributes], dim=-1)
        attribute_mask = {}
        for head in probs:
            mask = torch.zeros(*probs[head].shape, dtype=torch.bool).to(probs[head].device)
            if self.output_distributions[head].conditioning_transform == "mask":
                for i, hc in enumerate(self.output_distributions[head].condition_on):
                    if self.output_distributions[hc].domain == "node":
                        feat_modes = {hc: torch.argmax(probs[hc], dim=-1) for hc in
                                      self.output_distributions[head].condition_on}
                        masked_attributes = self.output_distributions[head].masked_attributes[i]
                        distribution_attributes = list(self.output_distributions[hc].attributes)
                        masked_attributes_idx = torch.tensor([distribution_attributes.index(attr)
                                                              for attr in masked_attributes]).to(probs[hc].device)
                        mask_layer = ((torch.isin(feat_modes[hc], masked_attributes_idx)) & (probs[hc].sum(dim=-1) != 0)).unsqueeze(-1)
                        mask = mask | mask_layer
                    elif self.output_distributions[hc].domain == "nodeset":
                        feat_modes = {hc: torch.argmax(probs[hc].squeeze(-1), dim=-1) for hc in
                                      self.output_distributions[head].condition_on}
                        ids = torch.arange(feat_modes[hc].shape[0], device=feat_modes[hc].device)
                        mask[ids, feat_modes[hc]] = True
            else:
                pass
            attribute_mask[head] = ~mask

        return attribute_mask

    def compute_attribute_mask_from_decoder_output(self, logits:Dict[str, torch.Tensor]=None,
                                                   probs:Dict[str, torch.Tensor]=None):

        """
        Computes the attribute mask for each head in the output distribution. At the moment this is exclusive to the
        cave escape environment.

        The attribute mask is a boolean tensor indicating which logits/probs of the output distribution should be kept
        during decoding.

        Masking logic:
            1. compute a mask that filters for p(nav) > 0.5
            2. compute a mask to sample layout nodes according to p({moss, empty}|nav) and p({lava, wall}|~nav)
            3. goal will be sampled from p(goal|nav)
            4. start will be sampled from p(start|nav,~goal)

        Args:
            logits (Dict[str, torch.Tensor], optional): Dictionary of logits for each head in the output distribution.
                The shape of each tensor should be (*, B, max_nodes, d), where * is any number of additional dimensions,
                B is the batch size, max_nodes is the maximum number of nodes in the graph, and d is the dimensionality
                of the output distribution. Either `logits` or `probs` should be provided.
            probs (Dict[str, torch.Tensor], optional): Dictionary of probabilities for each head in the output
                distribution. The shape of each tensor should be the same as for `logits`. Either `logits` or `probs`
                should be provided.

        Returns:
            attribute_mask (Dict[str, torch.Tensor]): Dictionary of attribute masks for each head in the output
                distribution. The shape of each tensor is (*, B, max_nodes, d), where * is any number of additional
                dimensions, B is the batch size, max_nodes is the maximum number of nodes in the graph, and d is the
                dimensionality of the output distribution.
        """

        if probs is None:
            assert logits is not None, "Must provide either Fx, logits or probs"
            probs = self.param_p(logits=logits)
        else:
            assert logits is None, "Cannot specify both logits and probs"

        # 1. compute a mask that filters for p(nav) > 0.5
        nav_attributes_idx = [i for i, attr in enumerate(self.output_distributions['layout'].attributes)
                          if attr in ['moss', 'empty']]
        nav_probs = torch.stack([probs['layout'][..., i] for i in nav_attributes_idx], dim=-1).sum(dim=-1)
        nav_mask = torch.zeros_like(nav_probs, dtype=torch.bool)
        nav_mask[nav_probs > 0.5] = True

        # 2. compute a mask to sample layout nodes according to p({moss, empty}|nav) and p({lava, wall}|~nav)
        attribute_mask = {'layout': torch.zeros_like(probs['layout'], dtype=torch.bool)}
        for i in range(len(self.output_distributions['layout'].attributes)):
            if i in nav_attributes_idx:
                attribute_mask['layout'][..., i] = nav_mask
            else:
                attribute_mask['layout'][..., i] = ~nav_mask
        # 3. goal will be sampled from p(goal|nav)
        attribute_mask['goal_location'] = nav_mask.unsqueeze(-1)
        # 4. start will be sampled from p(start|nav,~goal)
        attribute_mask['start_location'] = attribute_mask['goal_location'].clone()
        probs['goal_location'] = probs['goal_location'] * nav_mask.unsqueeze(-1)
        goal_ids = torch.argmax(probs['goal_location'].squeeze(-1), dim=-1)
        batch_ids = torch.arange(goal_ids.shape[0]).to(goal_ids.device)
        attribute_mask['start_location'][batch_ids, goal_ids] = False

        return attribute_mask

    def masked_logits(self, logits: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Masks the given logits with the provided binary mask

        Args:
            logits (Dict[str, torch.Tensor]): The input logits to be masked
            mask (Dict[str, torch.Tensor]): The binary mask to be applied on the logits

        Returns:
            masked_logits (Dict[str, torch.Tensor]): The masked logits
        """
        masked_logits = {}
        for head in logits:
            masked_logits[head] = self.mask_logits(logits[head], mask[head])
        return masked_logits

    def mask_logits(self, logits: torch.Tensor, mask: torch.Tensor, value=float('-inf')) -> torch.Tensor:
        """
        Masks logits according to a boolean mask.

        Args:
            logits (Tensor): Logits tensor of shape (batch_size, num_nodes, num_classes)
            mask (Tensor): Boolean mask tensor of shape (batch_size, num_nodes, num_classes)
            value (float, optional): Value to fill in the masked positions. Defaults to float('-inf').

        Returns:
            Tensor: Masked logits tensor of shape (batch_size, num_nodes, num_classes)
        """
        return logits.masked_fill(mask == 0, value)

    # Some extra methods for analysis

    def force_valid_masking(self, logits_Fx_masked: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
        """
        The force_valid_masking function modifies the masked logits such that any invalid logit entries are replaced with valid ones. Specifically, for any output distribution with a domain of "nodeset", it checks for any invalid logits by checking if all logit entries for a batch are equal to -inf. If any such invalid logits are detected, it selects a random node within the corresponding mask, and sets its logit entry to 1.0. If no valid nodes exist in the mask, it selects a random node from all possible nodes (assuming that the layout is then guaranteed to be invalid). The function returns the modified masked logits.

        Args:

            logits_Fx_masked (Dict[str, torch.Tensor]): A dictionary with keys corresponding to output distributions and values corresponding to masked logits (i.e., logits with invalid entries set to -inf).
            mask (Dict[str, torch.Tensor]): A dictionary with keys corresponding to output distributions and values corresponding to boolean masks indicating which entries in the original logits are valid.

        Returns:

            logits_Fx_masked (Dict[str, torch.Tensor]): A dictionary with keys corresponding to output distributions and values corresponding to modified masked logits, where any invalid entries have been replaced with valid ones.
        """

        for key in self.output_distributions:
            if self.output_distributions[key].domain == "nodeset":
                logits_invalid = torch.all(logits_Fx_masked[key].squeeze(-1) == float('-inf'), dim=-1)
                if logits_invalid.any():
                    invalid_batch_idx = logits_invalid.nonzero().flatten()
                    logger.warning(f"Invalid logits detected for {key} at "
                                   f"batch indices: {invalid_batch_idx.tolist()}."
                                   f" Sampling object at random within masked logits instead.")
                    sampled_idx = []
                    for idx in invalid_batch_idx:
                        sampling_set = torch.where(mask[key][idx])[0]
                        if len(sampling_set) == 0:
                            logger.warning(f"Invalid masks detected for {key} at "
                                           f"batch indices: {invalid_batch_idx.tolist()}."
                                           f" Sampling object at random within all nodes instead "
                                           f" (layout is guaranteed to be invalid).")
                            sampling_set = torch.randperm(logits_Fx_masked[key].shape[-2])
                            sampled_idx.append(sampling_set[0])
                            # a bit hacky, will only work for nav tasks
                            if key == "start_location" or key == "goal_location":
                                if key == "start_location":
                                    logits_Fx_masked["goal_location"][idx, sampling_set[1]] = 1.0
                                else:
                                    logits_Fx_masked["start_location"][idx, sampling_set[1]] = 1.0
                        else:
                            sampled_idx.append(sampling_set[torch.randint(low=0, high=len(sampling_set), size=(1,))])
                    logits_Fx_masked[key][invalid_batch_idx, sampled_idx] = 1.0  # can be set to any value > -inf

        return logits_Fx_masked

    def get_node_features(self, graph: Union[dgl.DGLGraph, List[dgl.DGLGraph]], node_attributes:List[str]=None,
                          reshape:bool=True, as_tensor=True, for_elbo=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Given a graph, returns the node features specified by the `node_attributes` list.

        Args:
            graph (Union[dgl.DGLGraph, List[dgl.DGLGraph]]): A DGLGraph or a list of DGLGraphs.
            node_attributes (List[str], optional): A list of node feature names to extract. If None, uses the default attributes.
            reshape (bool, optional): Whether to reshape the returned features to have an additional dimension. Defaults to True.
            as_tensor (bool, optional): Whether to return the features as a tensor or as a dictionary of tensors. Defaults to True.
            for_elbo (bool, optional): Whether to apply preprocessing to handle the ELBO loss computation. Defaults to False.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                - If `as_tensor` is True, returns a tensor of the extracted node features of dimensions (*, B, num_nodes, D)
                - If `as_tensor` is False, returns a dictionary of tensors of dimensions (*, B, num_nodes), where each key is a node feature name.

        Raises:
            AssertionError: If `as_tensor` and `for_elbo` flags are both True.
        """
        assert not (as_tensor and for_elbo), "Cannot use as_tensor and for_elbo flags at the same time."

        if node_attributes is None:
            node_attributes = self.attributes

        features, _ = util.get_node_features(graph, node_attributes=node_attributes, device=None,
                                                           reshape=reshape)

        if as_tensor:
            return features.float()
        else:
            features_dict = {}
            for i, key in enumerate(node_attributes):
                features_dict[key] = features[..., i].float()

            if for_elbo:
                # TODO URGENT: review and decide + this is not the best place to do this
                # 1. preprocess start / goal nodes as empty
                # features['empty'] += features['goal']
                # features['empty'] += features['start']

                # 2. preprocess start/goal nodes as max entropy (i.e. uniform distribution over wall/lava/moss/empty)

                # 3. preprocess start/goal nodes as max entropy over nav nodes (i.e. uniform distribution over moss/empty)

                # 4. add inductive biases about most likely node type "underneath/closeby"
                features_dict['moss'] += features_dict['goal']
                features_dict['empty'] += features_dict['start']

                # 5. ignore start/goal nodes in the loss computation (requires modification to the cross-entropy loss)
                # 6. marginalise cross-entropy loss over navigable nodes (i.e sum p(x = {moss, empty})) (requires modification to the cross-entropy loss)

            return features_dict

    def to_graph_features(self, logits:Dict[str, torch.Tensor]=None, probabilistic=False, masked=True, mask=None,
                          force_valid=True, sample=True) \
            -> Dict[str, torch.Tensor]:

        """
        Convert logits to graph features using the specified output distribution.

        Args:
            logits (Dict[str, torch.Tensor]): A dictionary containing the logits for each head of the model. If masked
                is True, then these logits must correspond to the attributes that will be used.
            probabilistic (bool):
                - If True, features will be a one-hot vector over their respective domain.
                - If False, features will be expressed as sampling probabilities \in [0,1] over their respective domain.
            masked (bool): If True, mask the logits prior to computing features.
            mask (torch.Tensor): A mask for the logits.
            force_valid (bool): If True, ensure that there is at least one valid option for each attribute.
            sample (bool): If True, attributes will be sampled instead of taking their mode. Only done if probabilistic
                is set to False.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the graph features.
        """

        if masked:
            if mask is None:
                if self.config.attribute_masking is None:
                    raise RuntimeError("Cannot do auto-masking when attribute masking set to None in decoder config.")
                mask = self.compute_attribute_mask_from_decoder_output(logits=logits)
            logits = self.masked_logits(logits, mask=mask)
            if force_valid:
                logits = self.force_valid_masking(logits, mask=mask)
        else:
            if mask is not None:
                raise ValueError("Mask provided but masked=False")

        if probabilistic:
            param = self.param_p(logits=logits)
            sample = False
        else:
            if sample:
                param = self.sample(logits=logits, num_samples=1)
                for head in param:
                    param[head] = param[head].squeeze(0)
            else:
                param = self.param_m(logits=logits)

        feats = {}
        for head in param:
            if head == "adjacency":
                continue
            if self.output_distributions[head].family == "one_hot_categorical":
                if self.output_distributions[head].domain == "node":
                    for i, attr in enumerate(self.output_distributions[head].attributes):
                        feats[attr] = param[head][..., i]
                elif self.output_distributions[head].domain == "nodeset":
                    assert len(self.output_distributions[head].attributes) == 1, \
                        "Only one attribute can be specified for a nodeset distribution"
                    attr = self.output_distributions[head].attributes[0]
                    feats[attr] = param[head].squeeze(-1)
                else:
                    raise NotImplementedError(f"Domain {self.output_distributions[head].domain} not implemented.")
            else:
                raise NotImplementedError(f"Family {self.output_distributions[head].family} not implemented.")

        assert len(feats) == len(self.attributes), "Number of features does not match number of attributes"
        assert all([feats[attr].shape == feats[self.attributes[0]].shape for attr in feats]), \
            "Features must have the same shape"
        return feats

    def to_graph(self, logits:Dict[str, torch.Tensor], dim_gridgraph:tuple=None, make_batch:bool=False,
                 edge_config=None, masked=True, mask=None, force_valid=True, sample=True) \
            -> Union[List[dgl.DGLGraph], dgl.DGLGraph]:
        """
        Converts a dictionary of logits to a list of DGLGraph objects / a batched DGLGraph representing the navigation environment.

        Args:
        - logits: A dictionary of logits from which to extract graph features.
        - dim_gridgraph: A tuple containing the dimensions of the 2D gridworld used to represent the navigation environment. If not specified, default dimensions are used.
        - make_batch: A boolean indicating whether to convert the graphs to a batch format or not. Default is False.
        - edge_config: A dictionary specifying the type and/or number of edges for each graph. Default is None.
        - masked: A boolean indicating whether to mask the logits before extracting features. Default is True.
        - mask: A tensor mask to use when masking the logits. Default is None.
        - force_valid: A boolean indicating whether to force valid masking when masking the logits. Default is True.
        - sample (bool): If True, attributes will be sampled instead of taking their mode. Only done if probabilistic
        is set to False.

        Returns:
        - graphs: A list of dense DGLGraph objects representing the navigation environment. If make_batch is True, a single batched DGLGraph object is returned instead.
        """
        if self.shared_params.data_encoding == "dense":
            return self._to_dense_graph(logits, dim_gridgraph=dim_gridgraph, make_batch=make_batch,
                                        edge_config=edge_config, masked=masked, mask=mask, force_valid=force_valid,
                                        sample=sample)
        elif self.shared_params.data_encoding == "minimal":
            return self._to_minimal_graph(logits)
        else:
            raise NotImplementedError(f"to_graph() not implemented for {self.shared_params.data_encoding} encoding.")

    def _to_dense_graph(self,
                        logits:Dict[str, torch.Tensor],
                        dim_gridgraph:tuple=None,
                        make_batch: bool = False,
                        edge_config=None,
                        masked=True,
                        mask=None,
                        force_valid=True,
                        sample=True) \
            -> Union[List[dgl.DGLGraph], dgl.DGLGraph]:
        """
        Converts a dictionary of logits to a list of dense DGLGraph objects / a batched DGLGraph representing the navigation environment.

        Args:
        - logits: A dictionary of logits from which to extract graph features.
        - dim_gridgraph: A tuple containing the dimensions of the 2D gridworld used to represent the navigation environment. If not specified, default dimensions are used.
        - make_batch: A boolean indicating whether to convert the graph to a batch format or not. Default is False.
        - edge_config: A dictionary specifying the type and/or number of edges for the graph. Default is None.
        - masked: A boolean indicating whether to mask the logits before extracting features. Default is True.
        - mask: A tensor mask to use when masking the logits. Default is None.
        - force_valid: A boolean indicating whether to force valid masking when masking the logits. Default is True.
        - sample (bool): If True, attributes will be sampled instead of taking their mode. Only done if probabilistic
        is set to False.
        Returns:
        - graphs: A list of dense DGLGraph objects representing the navigation environment. If make_batch is True, a single batched DGLGraph object is returned instead.
        """

        Fx = self.to_graph_features(logits=logits, probabilistic=False, masked=masked, mask=mask,
                                    force_valid=force_valid, sample=sample)

        if dim_gridgraph is None:
            dim_gridgraph = tuple([x - 2 for x in self.shared_params.gridworld_data_dim[1:]])

        assert len(dim_gridgraph) == 2, "Only 2D Gridworlds are currently supported"

        graphs, _ = tr.Nav2DTransforms.features_to_dense_graph(Fx, dim_gridgraph, edge_config=edge_config,
                                                   to_dgl=True, make_batch=make_batch)

        return graphs

    def _to_minimal_graph(self, logits:Dict[str, torch.Tensor]) -> dgl.DGLGraph:

        raise NotImplementedError()

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
                                 permutations=self.permutations,
                                 output_keys=self.hparams.config_model.model.outputs)
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