from typing import List, Any

import torch
import dgl
import networkx as nx
import numpy as np

from data_generators import Batch

def shortest_paths(graph:nx.Graph, source:int, target:int, num_paths:int=1) -> List[Any]:
    """
    Compute shortest paths from source to target in graph.
    """
    assert num_paths >= 1

    #graph = nx.Graph(graph)

    if not nx.has_path(graph, source, target):
        return []

    if num_paths == 1:
        return [nx.shortest_path(graph, source, target, method="dijkstra")]
    else:
        return [p for p in nx.all_shortest_paths(graph, source, target, method='dijkstra')][:num_paths]

def shortest_path_length(graph:nx.Graph, source:int, target:int) -> int:
    """
    Compute shortest path length from source to target in graph.
    """
    #graph = nx.Graph(graph)
    return nx.shortest_path_length(graph, source, target)
    #return len(shortest_paths(graph, source, target, num_paths=1)[0])

def resistance_distance(graph:nx.Graph, source:int, target:int) -> int:
    """
    Compute resistance distance from source to target in graph. Graph must be strongly connectected (1 single component)
    """
    #graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan
    else:
       #component = graph.subgraph(nx.node_connected_component(graph, source))
        return nx.resistance_distance(graph, source, target)

def active_node_count(graph:nx.Graph) -> int:
    """
    Compute active node count in graph.
    """
    #graph = nx.Graph(graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.number_of_nodes(graph)

def len_connected_component(graph:nx.Graph, source, target) -> int:

    #graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan

    return len(nx.node_connected_component(graph, source))

def is_solvable(graph:nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target)

def valid_start_goal(start_nodes, goal_nodes, A_red):
    """
    Check if the start and goal nodes are valid.
    """

    batch_inds = torch.arange(0, start_nodes.shape[0])

    mask1 = start_nodes == goal_nodes
    mask2 = A_red[batch_inds, start_nodes].sum(dim=-1) == 0
    mask3 = A_red[batch_inds, goal_nodes].sum(dim=-1) == 0
    # valid only if NOT(start==goal AND no edges from start AND no edges from goal)
    valid = ~(mask1 & mask2 & mask3)
    #TODO: check with this
    # check that start and goal are not isolated (inactive) nodes
    #if start in list(nx.isolates(graph)) or goal in list(nx.isolates(graph)):
        #return False

    return valid

def check_validity_and_convert_to_graph(mode_A, mode_Fx, start_dim, goal_dim, check_validity=True, correct_A=False, correct_Fx=False):
    """
    Check if the graph is valid.
    """

    #TODO correct_Fx
    mode_A = mode_A.reshape(mode_A.shape[0], -1, 2)
    n_nodes = mode_Fx.shape[1]

    start_nodes = mode_Fx[...,start_dim].argmax(dim=-1)
    goal_nodes = mode_Fx[...,goal_dim].argmax(dim=-1)

    if check_validity:
        mode_A, valid_A = Batch.is_valid_reduced_A(mode_A, n_nodes, correct_A)
        valid_start_goal = valid_start_goal(start_nodes, goal_nodes, mode_A)
        valid = valid_A & valid_start_goal
    else:
        valid = None


    #TODO sort out force_valid
    graphs = Batch.encode_decoder_mode_to_graph(mode_A, mode_Fx, force_valid_A=False, device=mode_A.device)

    return graphs, start_nodes, goal_nodes, valid

def compute_metrics(mode_A, mode_Fx, desired_metrics:List[str]=
    ["valid","solvable","shortest_path", "resistance", "navigable_nodes"], start_dim=2, goal_dim=3) -> dict:
    """
    :param graphs: List[DGLGraph]
    :param desired_metrics: List[str]
    :param start_dim: int
    :param goal_dim: int
    :return: metrics: Dict[str, List[float]]
    """

    graphs, start_nodes, goal_nodes, valid_mask = check_validity_and_convert_to_graph(mode_A, mode_Fx, start_dim, goal_dim)

    metrics = {k: [] for k in desired_metrics}
    try:
        metrics["valid"] = valid_mask.tolist()
    except KeyError:
        pass

    for i, graph in enumerate(graphs):
        if isinstance(graph, dgl.DGLGraph):
            graph = graph.to_networkx()
            graph = nx.Graph(graph)
        else:
            raise ValueError("graph must be a DGLGraph")

        solvable = False
        computed_m = []

        for metric in desired_metrics:
            if metric == "solvable":
                try:
                    valid = metrics["valid"][i]
                except KeyError:
                    valid = True
                if valid:
                    start_node = start_nodes[i]
                    goal_node = goal_nodes[i]
                    solvable = is_solvable(graph, start_node, goal_node)
                else:
                    solvable = False
                metrics[metric].append(solvable)
                computed_m.append(metric)
            else:
                if "solvable" in computed_m and not solvable:
                    metrics["shortest_path"].append(np.nan)
                    metrics["resistance"].append(np.nan)
                    metrics["navigable_nodes"].append(np.nan)
                    computed_m.extend(["shortest_path", "resistance", "navigable_nodes"])
                    break
                else:
                    graph = graph.subgraph(nx.node_connected_component(graph, start_node))
                    if metric == "shortest_path":
                        metrics[metric].append(shortest_path_length(graph, start_node, goal_node))
                        computed_m.append(metric)
                    elif metric == "resistance":
                        metrics[metric].append(resistance_distance(graph, start_node, goal_node))
                        computed_m.append(metric)
                    elif metric == "navigable_nodes":
                        metrics[metric].append(len_connected_component(graph, start_node, goal_node))
                        computed_m.append(metric)

    return metrics