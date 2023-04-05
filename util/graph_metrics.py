import logging

from typing import List, Any, Dict, Union, Iterable, Tuple

import torch
import dgl
import networkx as nx
import numpy as np

import maze_representations.util.transforms as tr
import maze_representations.util.util as util

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

def num_nav_nodes(graph: nx.Graph) -> int:
    """
    Compute navigable node count in graph. Input graph should have a grid
    Inputs:
    - graph: graph (nx.Graph)
    - start: starting node (int)
    Output: number of navigable nodes (int)
    """
    nav = len([graph.degree[i] for i in graph.nodes if graph.degree[i] != 0])
    return nav


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


def compute_metrics(graphs: Union[List[dgl.DGLGraph], List[nx.Graph]], labels=None) -> Dict[str, torch.Tensor]:

    if isinstance(graphs[0], dgl.DGLGraph):
        graphs = [util.dgl_to_nx(graph) for graph in graphs]

    metrics = {"valid": [], "solvable": [], "shortest_path": [], "resistance": [], "navigable_nodes":[]}
    for i, graph in enumerate(graphs):
        start_node = [n for n in graph.nodes if graph.nodes[n]['start'] == 1.0]
        goal_node = [n for n in graph.nodes if graph.nodes[n]['goal'] == 1.0]
        assert len(start_node) == 1
        assert len(goal_node) == 1
        start_node = start_node[0]
        goal_node = goal_node[0]
        if start_node != goal_node:
            metrics["valid"].append(True)
            solvable = nx.has_path(graph, start_node, goal_node)
        else:
            metrics["valid"].append(False)
            solvable = False
            # assert metrics["valid"][i] == False
        if not solvable:  # then these metrics do not make sense
            metrics["solvable"].append(False)
            metrics["shortest_path"].append(np.nan)
            metrics["resistance"].append(np.nan)
            metrics["navigable_nodes"].append(np.nan)
        else:
            subg = graph.subgraph(nx.node_connected_component(graph, start_node))
            metrics["solvable"].append(True)
            metrics["shortest_path"].append(shortest_path_length(subg, start_node, goal_node))
            metrics["resistance"].append(resistance_distance(subg, start_node, goal_node))
            metrics["navigable_nodes"].append(num_nav_nodes(subg))

    for metric in metrics:
        if metric in ["valid", "solvable"]:
            metrics[metric] = torch.tensor(metrics[metric], dtype=torch.bool)
        elif metric in ["shortest_path", "resistance", "navigable_nodes"]:
            metrics[metric] = torch.tensor(metrics[metric], dtype=torch.float)
        else:
            raise ValueError("Unknown metric")

    return metrics


def get_non_nav_spl(non_nav_nodes: List[Tuple[int, int]], spl_nav: Dict[Tuple[int, int], int],
                    grid_size: Tuple[int, int], depth: int = 3) -> Dict[Tuple[int, int], int]:
    neighbors_of_non_nav_nodes = get_neighbors(non_nav_nodes, list(spl_nav.keys()), grid_size)
    shortest_path_lengths = {}
    for node_ind, neighbors in enumerate(neighbors_of_non_nav_nodes):
        neighbors = [n for n in neighbors if n in spl_nav]
        if neighbors:
            pathlength = int(np.min([spl_nav[neighbor] for neighbor in neighbors])) + 1
        else:
            pathlength = None  # pathlength = np.max(list(shortest_path_lengths_nav.values())) + 1
        shortest_path_lengths[non_nav_nodes[node_ind]] = pathlength
    border_nodes = [n for n in shortest_path_lengths if shortest_path_lengths[n] is not None]
    nodes_to_remove = []
    for node in shortest_path_lengths:
        if shortest_path_lengths[node] is None:
            grid_graph = nx.grid_2d_graph(*grid_size)
            spl_ = dict(nx.single_target_shortest_path_length(grid_graph, node))
            if np.min([spl_[border_node] for border_node in border_nodes]) > depth:
                nodes_to_remove.append(node)
            else:
                spl_goal = [spl_[border_node] + shortest_path_lengths[border_node] for border_node in border_nodes]
                pathlength = np.min(spl_goal)
                shortest_path_lengths[node] = pathlength
    [shortest_path_lengths.pop(node) for node in nodes_to_remove]

    return shortest_path_lengths


def get_neighbors(nodes: List[Tuple[int, int]], neighbors_set: List[Tuple[int, int]],
                  grid_size: Tuple[int, int] = None):
    grid_graph = nx.grid_2d_graph(*grid_size)
    neighbors_grid = [list(grid_graph.neighbors(node)) for node in nodes]
    neighbors_grid = [list(set(neighbors_grid[i]) & set(neighbors_set)) for i in range(len(neighbors_grid))]
    return neighbors_grid
