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
import maze_representations.data_generators as dg
from util import DotDict


@pytest.fixture
def grid_nodes_6x6():
    nodes={
        (0,0):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (0,1):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (0,2):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (0,3):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (0,4):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (0,5):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,0):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,1):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,2):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,3):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,4):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (1,5):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,0):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,1):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,2):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,3):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,4):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (2,5):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,0):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,1):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,2):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,3):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,4):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (3,5):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,0):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,1):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,2):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,3):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,4):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (4,5):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,0):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,1):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,2):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,3):{'navigable':0.0,'empty':0.0,'non_navigable':1.0,'wall':1.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,4):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0},
        (5,5):{'navigable':1.0,'empty':1.0,'non_navigable':0.0,'wall':0.0,'lava':0.0,'moss':0.0,'start':0.0,'goal':0.0}
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


@pytest.fixture
def wfc_batch(grid_graph_6x6):
    with patch.object(dg.WaveCollapseBatch, '__init__', lambda self:None):
        b = dg.WaveCollapseBatch()
        b.dataset_meta = DotDict({'config':DotDict({})})
        b.dataset_meta.config.moss_distribution_params = DotDict({
                                                                     'score_metric':'shortest_path',
                                                                     'distribution':'power_rank', 'fraction':0.25,
                                                                     'temperature' :1, 'nodes':['empty']
                                                                     })
        # TODO: check correct
        b.dataset_meta.config.lava_distribution_params = DotDict({
                                                                     'score_metric':'shortest_path',
                                                                     'distribution':'power_rank', 'fraction':0.25,
                                                                     'temperature' :1, 'nodes':['wall']
                                                                     })
        b.dataset_meta.config.gridworld_data_dim = (3, 8, 8)
        b.dataset_meta.level_info = {
            'numpy':True, 'dtype':np.uint8, 'shape':(8, 8, 3)
            }
        b.dataset_meta.config.graph_feature_descriptors = list(
            grid_graph_6x6.nodes[list(grid_graph_6x6.nodes)[0]].keys())

    return b


def test_place_goal(grid_graph_6x6):
    lg = [grid_graph_6x6]
    updated_g = copy.deepcopy(lg)
    spl = dg.WaveCollapseBatch._place_goal_random(updated_g)

    updated_g = updated_g[0]
    spl = spl[0]

    goal_node = [n for n in updated_g.nodes if updated_g.nodes[n]['goal'] == 1.0]
    assert len(goal_node) == 1
    goal_node = goal_node[0]
    for attr in updated_g.nodes[goal_node]:
        if attr in ['goal', 'navigable']:
            assert updated_g.nodes[goal_node][attr] == 1.0
        else:
            assert updated_g.nodes[goal_node][attr] == 0.0

    for node in spl:
        if updated_g.nodes[node]['goal'] == 1.0:
            assert spl[node] == 0
        elif updated_g.nodes[node]['navigable'] == 1.0:
            assert spl[node] > 0
        else:
            assert False, "shortest path length should only be calculated for navigable nodes"

    return updated_g, spl


def test_place_moss(grid_graph_6x6, wfc_batch):
    gl = [grid_graph_6x6]
    b = wfc_batch
    spl = dg.WaveCollapseBatch._place_goal_random(gl)
    gl_orig = copy.deepcopy(gl)
    probs = b._place_moss_cave_escape(gl, spl)

    g = gl_orig[0]
    updated_g = gl[0]
    probs = probs[0]

    nav_nodes = [n for n in updated_g.nodes if updated_g.nodes[n]['navigable'] == 1.0]
    moss_nodes = [n for n in updated_g.nodes if updated_g.nodes[n]['moss'] == 1.0]
    goal_node = [n for n in updated_g.nodes if updated_g.nodes[n]['goal'] == 1.0]
    assert len(goal_node) == 1
    goal_node = goal_node[0]

    sum_p = 0
    for node in nav_nodes:
        if node in moss_nodes:
            assert g.nodes[node]['moss'] == 0.0
            assert g.nodes[node]['navigable'] == 1.0
            assert g.nodes[node]['empty'] == 1.0
            assert updated_g.nodes[node]['navigable'] == 1.0
            assert updated_g.nodes[node]['empty'] == 0.0

        assert probs[node] <= 1.0
        if node == goal_node:
            assert probs[node] == 0.0
        else:
            assert probs[node] > 0.0
        sum_p += probs[node]

    assert sum_p == pytest.approx(1.0)
    assert np.array(list(probs.values())).sum() == pytest.approx(1.0)

    return updated_g


def test_place_lava(grid_graph_6x6, wfc_batch):
    gl = [grid_graph_6x6]
    b = wfc_batch
    spl = dg.WaveCollapseBatch._place_goal_random(gl)
    _ = b._place_moss_cave_escape(gl, spl)
    gl_orig = copy.deepcopy(gl)
    probs = b._place_lava_cave_escape(gl, spl)

    g = gl_orig[0]
    updated_g = gl[0]
    probs = probs[0]

    non_nav_nodes = [n for n in updated_g.nodes if updated_g.nodes[n]['non_navigable'] == 1.0]
    lava_nodes = [n for n in updated_g.nodes if updated_g.nodes[n]['lava'] == 1.0]
    sum_p = 0
    for node in non_nav_nodes:
        if node in lava_nodes:
            assert g.nodes[node]['lava'] == 0.0
            assert g.nodes[node]['non_navigable'] == 1.0
            assert g.nodes[node]['wall'] == 1.0
            assert updated_g.nodes[node]['non_navigable'] == 1.0
            assert updated_g.nodes[node]['empty'] == 0.0
        assert probs[node] > 0.0
        assert probs[node] <= 1.0
        sum_p += probs[node]

    assert sum_p == pytest.approx(1.0)
    assert np.array(list(probs.values())).sum() == pytest.approx(1.0)

    return updated_g


def test_place_start(grid_graph_6x6, wfc_batch):
    gl = [grid_graph_6x6]
    b = wfc_batch
    spl = dg.WaveCollapseBatch._place_goal_random(gl)
    _ = b._place_moss_cave_escape(gl, spl)
    _ = b._place_lava_cave_escape(gl, spl)
    gl_orig = copy.deepcopy(gl)
    possible_starts = b._place_start_cave_escape(gl, spl)

    g = gl_orig[0]
    updated_g = gl[0]
    possible_starts = possible_starts[0]

    start_nodes = [n for n in updated_g.nodes if updated_g.nodes[n]['start'] == 1.0]
    assert len(start_nodes) == 1
    assert len(possible_starts) >= 1
    start_node = start_nodes[0]
    assert start_node in possible_starts
    for attr in updated_g.nodes[start_node]:
        if attr in ['start', 'navigable']:
            assert updated_g.nodes[start_node][attr] == 1.0
        else:
            assert updated_g.nodes[start_node][attr] == 0.0
    for sn in possible_starts:
        assert g.nodes[sn]['goal'] == 0
        assert g.nodes[sn]['navigable'] == 1
        assert g.nodes[sn]['empty'] == 1 or g.nodes[sn]['moss'] == 1
        assert g.nodes[sn]['wall'] == 0
        assert g.nodes[sn]['lava'] == 0

    goal_node = [n for n in updated_g.nodes if updated_g.nodes[n]['goal'] == 1.0]
    assert len(goal_node) == 1
    goal_node = goal_node[0]
    assert nx.has_path(updated_g, start_node, goal_node)

    return updated_g

    test_add_edges_stage1(grid_graph_6x6, wfc_batch)
    g, edge_g = test_add_edges_stage2(g, wfc_batch)
    test_update_graph_features(g, edge_g, wfc_batch)


def test_add_edges_stage1(grid_graph_6x6, wfc_batch):
    g = grid_graph_6x6
    b = wfc_batch
    edge_config = DotDict({'navigable':{'between':['navigable'], 'structure':'grid', 'weight':None}})
    g_test = nx.create_empty_copy(g, with_data=True)
    edge_g = b._add_edges([g_test], edge_config)
    edge_g = edge_g[0]
    assert set(g_test.edges()) == set(g.edges())
    assert set(g_test.edges()) == set(edge_g['navigable'].edges())
    nav_nodes = [n for n in g.nodes if g.nodes[n]['navigable'] == 1.0]
    for n in nav_nodes:
        for m in nav_nodes:
            assert nx.has_path(g_test, n, m)


def test_add_edges_stage2(grid_graph_6x6, wfc_batch):
    gl = [grid_graph_6x6]
    b = wfc_batch
    spl = dg.WaveCollapseBatch._place_goal_random(gl)
    _ = b._place_moss_cave_escape(gl, spl)
    _ = b._place_lava_cave_escape(gl, spl)
    _ = b._place_start_cave_escape(gl, spl)

    g = gl[0]

    edge_graphs = {'navigable':g}
    edge_config = DotDict({
                              'non_navigable':{'between':['non_navigable'], 'structure':'grid', 'weight':None},
                              'lava_goal'    :{'between':['lava', 'goal'], 'structure':None, 'weight':'lava_prob'},
                              'moss_goal'    :{'between':['moss', 'goal'], 'structure':None, 'weight':'moss_prob'},
                              'start_goal'   :{'between':['start', 'goal'], 'structure':None, 'weight':None}
                              })
    edge_g = b._add_edges([g], edge_config, [edge_graphs])
    edge_g = edge_g[0]
    for attr in edge_config:
        assert set(edge_g[attr].edges()).issubset(set(g.edges()))

    stacked_edges = set().union(*[set(edge_g[attr].edges()) for attr in edge_g])
    assert stacked_edges == set(g.edges())

    goal_node = [n for n in g.nodes if g.nodes[n]['goal'] == 1.0]
    assert len(goal_node) == 1
    goal_node = goal_node[0]
    neighbors_lava_goal = set([n for n in edge_g['lava_goal'].neighbors(goal_node)])
    lava_nodes = set([n for n in g.nodes if g.nodes[n]['lava'] == 1.0])
    assert neighbors_lava_goal == lava_nodes
    neighbors_moss_goal = set([n for n in edge_g['moss_goal'].neighbors(goal_node)])
    moss_nodes = set([n for n in g.nodes if g.nodes[n]['moss'] == 1.0])
    assert neighbors_moss_goal == moss_nodes
    neighbors_start_goal = set([n for n in edge_g['start_goal'].neighbors(goal_node)])
    start_node = set([n for n in g.nodes if g.nodes[n]['start'] == 1.0])
    assert neighbors_start_goal == start_node

    for n in g.nodes:
        if n != goal_node:
            if n not in lava_nodes:
                assert edge_g['lava_goal'].degree(n) == 0
            if n not in moss_nodes:
                assert edge_g['moss_goal'].degree(n) == 0
            if n not in start_node:
                assert edge_g['start_goal'].degree(n) == 0

    return g, edge_g


def test_update_graph_features(grid_graph_6x6, wfc_batch):
    gl = [grid_graph_6x6]
    b = wfc_batch
    spl = dg.WaveCollapseBatch._place_goal_random(gl)
    _ = b._place_moss_cave_escape(gl, spl)
    _ = b._place_lava_cave_escape(gl, spl)
    _ = b._place_start_cave_escape(gl, spl)
    edge_graphs = [{'navigable':gl[0]}]
    edge_config = DotDict({
                              'non_navigable':{'between':['non_navigable'], 'structure':'grid', 'weight':None},
                              'lava_goal'    :{'between':['lava', 'goal'], 'structure':None, 'weight':'lava_prob'},
                              'moss_goal'    :{'between':['moss', 'goal'], 'structure':None, 'weight':'moss_prob'},
                              'start_goal'   :{'between':['start', 'goal'], 'structure':None, 'weight':None}
                              })

    edge_g = b._add_edges(gl, edge_config, edge_graphs)
    g = gl[0]
    edge_g = edge_g[0]

    b._update_graph_features(graphs=[edge_g], reference_graphs=[g])
    for n in g.nodes:
        for key in edge_g:
            for attr in g.nodes[n]:
                assert g.nodes[n][attr] == edge_g[key].nodes[n][attr]
