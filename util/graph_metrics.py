from typing import List, Any

import torch
import dgl
import networkx as nx
import numpy as np


def shortest_paths(graph:nx.Graph, source:int, target:int, num_paths:int=1) -> List[Any]:
    """
    Compute shortest paths from source to target in graph.
    """
    assert num_paths >= 1

    graph = nx.Graph(graph)

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
    graph = nx.Graph(graph)
    return nx.shortest_path_length(graph, source, target)
    #return len(shortest_paths(graph, source, target, num_paths=1)[0])

def resistance_distance(graph:nx.Graph, source:int, target:int) -> int:
    """
    Compute resistance distance from source to target in graph.
    """
    graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan
    else:
        component = graph.subgraph(nx.node_connected_component(graph, source))
        return nx.resistance_distance(component, source, target)

def active_node_count(graph:nx.Graph) -> int:
    """
    Compute active node count in graph.
    """
    graph = nx.Graph(graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.number_of_nodes(graph)

def len_connected_component(graph:nx.Graph, source, target) -> int:

    graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan

    return len(nx.node_connected_component(graph, source))

def is_solvable(graph:nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target)

def is_valid(graph:nx.Graph, start, goal) -> bool:
    """
    Check if the graph is valid.
    """

    # check start and goal nodes are not placed in the same locaiton
    if start == goal:
        return False


    # check that start and goal are not isolated (inactive) nodes
    if start in list(nx.isolates(graph)) or goal in list(nx.isolates(graph)):
        return False

    return True

def compute_metrics(graphs:List[dgl.DGLGraph], desired_metrics:List[str]=
    ["valid","solvable","shortest_path", "resistance", "navigable_nodes"], start_dim=2, goal_dim=3):
    """
    :param graphs: List[DGLGraph]
    :param desired_metrics: List[str]
    :param start_dim: int
    :param goal_dim: int
    :return: metrics: Dict[str, List[float]]
    """

    metrics = {k: [] for k in desired_metrics}

    for graph in graphs:
        if isinstance(graph, dgl.DGLGraph):
            start_node = int(graph.ndata["feat"][:, start_dim].argmax())
            goal_node = int(graph.ndata["feat"][:, goal_dim].argmax())
            graph = graph.to_networkx()
        else:
            raise ValueError("graph must be either a DGLGraph or a nx.Graph")

        solvable = valid = False
        computed_m = []

        for metric in desired_metrics:
            if metric == "valid":
                valid = is_valid(graph, start_node, goal_node)
                metrics[metric].append(valid)
                computed_m.append(metric)
            elif metric == "solvable":
                if "valid" in computed_m and not valid:
                    solvable = False
                else:
                    solvable = is_solvable(graph, start_node, goal_node)
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