# unit-tests
#
# - on dataset graphs:
#  - test edges are OK
#  - test graphmetrics computation
#  - R_Valid is broken

# TODO:
# -update rendering colors: dark red for walls, grey for empty space? (stone)
# - large scale dataset generation using the 1M dataset
# - update_edge_graph function to add nodes to the edged graphs
import copy

import pytest
from mock import patch
import numpy as np
import networkx as nx
import dgl
from util import DotDict
from collections import defaultdict

from maze_representations.util.graph_metrics import *


@pytest.fixture
def grid_nodes_6x6():
    nodes={
        (0, 0)       :{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (0, 1):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (0, 2):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':0.0, 'lava':1.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (0, 3):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':1.0,
            'goal'     :0.0
            }, (0, 4):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (0, 5):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 0):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 1):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 2):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 3):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 4):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (1, 5):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 0):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 1):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 2):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 3):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 4):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (2, 5):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (3, 0):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (3, 1):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (3, 2):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (3, 3):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :1.0
            }, (3, 4):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (3, 5):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 0):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 1):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 2):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 3):{
            'navigable':1.0, 'empty':0.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':1.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 4):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (4, 5):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':0.0, 'lava':1.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 0):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 1):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 2):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 3):{
            'navigable':0.0, 'empty':0.0, 'non_navigable':1.0, 'wall':1.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 4):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }, (5, 5):{
            'navigable':1.0, 'empty':1.0, 'non_navigable':0.0, 'wall':0.0, 'lava':0.0, 'moss':0.0, 'start':0.0,
            'goal'     :0.0
            }
        }

    return nodes


@pytest.fixture
def grid_graph_6x6(grid_nodes_6x6):
    nodes = grid_nodes_6x6
    edges=[((0,0),(1,0)),((0,0),(0,1)),((0,1),(1,1)),((0,3),(1,3)),((0,3),(0,4)),((0,4),(1,4)),((0,4),(0,5)),
           ((0,5),(1,5)),((1,0),(2,0)),((1,0),(1,1)),((1,1),(2,1)),((1,1),(1,2)),((1,2),(2,2)),((1,2),(1,3)),
           ((1,3),(2,3)),((1,3),(1,4)),((1,4),(2,4)),((1,4),(1,5)),((2,0),(2,1)),((2,1),(3,1)),((2,1),(2,2)),
           ((2,2),(3,2)),((2,2),(2,3)),((2,3),(3,3)),((2,3),(2,4)),((3,1),(4,1)),((3,1),(3,2)),((3,2),(3,3)),
           ((3,3),(4,3)),((4,0),(5,0)),((4,0),(4,1)),((4,3),(4,4)),((4,4),(5,4)),((5,4),(5,5))]

    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    nx.set_node_attributes(G, nodes)
    G.add_edges_from(edges)

    return G


def test_compute_metrics(grid_graph_6x6):
    g = nx.convert_node_labels_to_integers(grid_graph_6x6)
    g = dgl.from_networkx(g, node_attrs=g.nodes[list(g.nodes)[0]].keys())
    metrics = defaultdict(list)
    compute_metrics([g], metrics)

    assert metrics['valid'][0]
    assert metrics['solvable'][0]
    assert metrics['shortest_path'][0] == 3
    assert metrics['resistance'][0] == pytest.approx(1.749226)
    assert metrics['navigable_nodes'][0] == 26