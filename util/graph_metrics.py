from typing import List, Any

import torch
import dgl
import networkx as nx


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

def resistance_distance(graph:nx.Graph, source:int, target:int) -> int:
    """
    Compute resistance distance from source to target in graph.
    """
    graph = nx.Graph(graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.resistance_distance(graph, source, target)

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
        return 0

    return len(nx.node_connected_component(graph, source))

def is_solvable(graph:nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target)

def is_valid(graph:nx.Graph) -> bool:
    """
    Check if the graph is valid.
    """
    raise NotImplementedError