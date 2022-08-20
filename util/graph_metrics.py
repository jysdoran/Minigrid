from typing import List, Any, Dict, Union, Iterable

import torch
import dgl
import networkx as nx
import numpy as np


def shortest_paths(graph: nx.Graph, source: int, target: int, num_paths: int = 1) -> List[Any]:
    """
    Compute shortest paths from source to target in graph.
    """
    assert num_paths >= 1

    # graph = nx.Graph(graph)

    if not nx.has_path(graph, source, target):
        return []

    if num_paths == 1:
        return [nx.shortest_path(graph, source, target, method="dijkstra")]
    else:
        return [p for p in nx.all_shortest_paths(graph, source, target, method='dijkstra')][:num_paths]


def shortest_path_length(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute shortest path length from source to target in graph.
    """
    # graph = nx.Graph(graph)
    return nx.shortest_path_length(graph, source, target)
    # return len(shortest_paths(graph, source, target, num_paths=1)[0])


def resistance_distance(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute resistance distance from source to target in graph. Graph must be strongly connectected (1 single component)
    """
    # graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan
    else:
        # component = graph.subgraph(nx.node_connected_component(graph, source))
        return nx.resistance_distance(graph, source, target)


def active_node_count(graph: nx.Graph) -> int:
    """
    Compute active node count in graph.
    """
    # graph = nx.Graph(graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.number_of_nodes(graph)


def len_connected_component(graph: nx.Graph, source, target) -> int:
    # graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        return np.nan

    return len(nx.node_connected_component(graph, source))


def is_solvable(graph: nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target)


def prepare_graph(graph: Union[dgl.DGLGraph, nx.Graph], source: int, target: int):
    """Convert DGLGraph to nx.Graph and reduces it to a single component."""

    if isinstance(graph, dgl.DGLGraph):
        graph = graph.to_networkx()
    elif isinstance(graph, nx.Graph):
        pass
    else:
        raise ValueError("graph must be a DGLGraph or nx.Graph")
    graph = nx.Graph(graph)
    if not nx.has_path(graph, source, target):
        connected = False
    else:
        connected = True
    graph = graph.subgraph(nx.node_connected_component(graph, source))
    return graph, connected


def compute_metrics(graphs: Union[dgl.DGLGraph, nx.Graph],
                    metrics: Dict[str, List[float]],
                    start_nodes: Iterable[int], goal_nodes: Iterable[int]) -> Dict[str, List[float]]:
    """
     Compute nx specific metrics. Also returns the always solvable graph.
    """

    graphs_nx = []
    for i, graph in enumerate(graphs):
        start_node = int(start_nodes[i])
        goal_node = int(goal_nodes[i])
        try:
            solvable = metrics['solvable'][i]
        except IndexError:
            graph, solvable = prepare_graph(graph, start_node, goal_node)
            metrics["solvable"].append(solvable)
        graphs_nx.append(graph)
        if not solvable:  # then this metrics do not make sense
            metrics["shortest_path"].append(np.nan)
            metrics["resistance"].append(np.nan)
            metrics["navigable_nodes"].append(np.nan)
        else:
            metrics["shortest_path"].append(shortest_path_length(graph, start_node, goal_node))
            metrics["resistance"].append(resistance_distance(graph, start_node, goal_node))
            metrics["navigable_nodes"].append(graph.number_of_nodes())  # already reduced to principal component

    return metrics, graphs_nx
