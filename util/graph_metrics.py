import logging

from typing import List, Any, Dict, Union, Iterable, Tuple

import torch
import dgl
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def shortest_paths(graph: nx.Graph, source: int, target: int, num_paths: int = 1) -> List[Any]:
    """
    Compute shortest paths from source to target in graph.
    """
    assert num_paths >= 1

    # graph = nx.Graph(graph)

    if source == target or not nx.has_path(graph, source, target):
        return []

    if num_paths == 1:
        return [nx.shortest_path(graph, source, target, method="dijkstra")]
    else:
        return [p for p in nx.all_shortest_paths(graph, source, target, method='dijkstra')][:num_paths]


def shortest_path_length(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute shortest path length from source to target in graph.
    """
    return nx.shortest_path_length(graph, source, target)


def resistance_distance(graph: nx.Graph, source: int, target: int) -> int:
    """
    Compute resistance distance from source to target in graph. Graph must be strongly connected (1 single component)
    """
    if source == target or not nx.has_path(graph, source, target):
        return np.nan
    else:
        return nx.resistance_distance(graph, source, target)


def active_node_count(graph: nx.Graph) -> int:
    """
    Compute active node count in graph.
    """
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.number_of_nodes(graph)


def len_connected_component(graph: nx.Graph, source, target) -> int:
    if not source == target or nx.has_path(graph, source, target):
        return np.nan

    return len(nx.node_connected_component(graph, source))


def is_solvable(graph: nx.Graph, source, target) -> bool:
    """
    Check if the graph is solvable.
    """
    return nx.has_path(graph, source, target) and source != target


def prepare_graph(graph: Union[dgl.DGLGraph, nx.Graph], source: int=None, target: int=None)\
        -> Tuple[nx.Graph, bool, bool]:
    """Convert DGLGraph to nx.Graph and reduces it to a single component. Containing the source node.
    If the source node is not specified, the largest component is returned."""

    if isinstance(graph, dgl.DGLGraph):
        graph = dgl.to_networkx(graph.cpu(), node_attrs=graph.ndata.keys())
    elif isinstance(graph, nx.Graph):
        pass
    else:
        raise ValueError("graph must be a DGLGraph or nx.Graph")
    graph = nx.Graph(graph)
    inactive_nodes = [x for x, y in graph.nodes(data=True) if y['navigable'] < .5]
    graph.remove_nodes_from(inactive_nodes)
    nodes = set(graph.nodes)

    if source is not None:
        # Catch any unwanted exceptions here, but it should be handled if the graph supplied is valid
        if len(nodes) < 2 or source == target or source not in nodes or target not in nodes:
            valid = False
            connected = False
            return graph, valid, connected
        else:
            valid = True

        # if graph.degree[source] == 0 or graph.degree[target] == 0:
        #     valid = False
        # else:
        #     valid = True

        if nx.has_path(graph, source, target):
            connected = True
        else:
            connected = False

        component = nx.node_connected_component(graph, source)
    else:
        components = [graph.subgraph(c).copy() for c in sorted(nx.connected_components(graph), key=len, reverse=True) if
                      len(c) > 1]
        component = components[0]
        valid = True
        connected = True

    graph = graph.subgraph(component)
    return graph, valid, connected


def compute_metrics(graphs: Union[dgl.DGLGraph, nx.Graph],
                    metrics: Dict[str, List[float]],
                    start_nodes: Iterable[int], goal_nodes: Iterable[int], labels=None) -> Dict[str, List[float]]:
    """
     Compute nx specific metrics. Also returns the always solvable graph.
     If the graph is not solvable, the metrics are set to np.nan.
     Assumes the metrics["valid"] as already been computed.
    """

    graphs_nx = []
    for i, graph in enumerate(graphs):
        start_node = int(start_nodes[i])
        goal_node = int(goal_nodes[i])
        graph, valid, solvable = prepare_graph(graph, start_node, goal_node)
        if valid != metrics["valid"][i]:
            if labels is None:
                logger.warning(f"compute_metrics() - Graph {i}/{len(graphs)} was marked valid:{valid}, which contradicts the metric provided valid:{metrics['valid'][i]}."
                               f"No labels were supplied.")
            else:
                logger.warning(
                    f"compute_metrics() - Graph (label:{labels[i]}) was marked valid:{valid}, which contradicts the metric provided valid:{metrics['valid'][i]}.")
        graphs_nx.append(graph)
        try:
            nodes = set(graph.nodes)
            assert start_node in nodes and goal_node in nodes
            _ = metrics['solvable'][i]
        except (IndexError, AssertionError) as e:
            if isinstance(e, AssertionError):
                solvable = False
            metrics["solvable"].append(solvable)
        if not solvable:  # then this metrics do not make sense
            metrics["shortest_path"].append(np.nan)
            metrics["resistance"].append(np.nan)
            metrics["navigable_nodes"].append(np.nan)
        else:
            metrics["shortest_path"].append(shortest_path_length(graph, start_node, goal_node))
            metrics["resistance"].append(resistance_distance(graph, start_node, goal_node))
            metrics["navigable_nodes"].append(graph.number_of_nodes())  # already reduced to principal component

    return metrics, graphs_nx
