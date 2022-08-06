from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch
import einops
from torchvision import transforms
import pickle
from pathlib import Path
import os
import dgl
from copy import deepcopy, copy

from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.solve.ShortestPaths import ShortestPaths
from mazelib.solve.ShortestPath import ShortestPath
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
from mazelib.solve.Chain import Chain
from mazelib.solve.RandomMouse import RandomMouse
from gym_minigrid.envs.multiroom_mod import MultiRoomEnv
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid as Minigrid_Grid
from gym_minigrid.minigrid import OBJECT_TO_IDX as Minigrid_OBJECT_TO_IDX
from gym_minigrid.minigrid import IDX_TO_OBJECT as Minigrid_IDX_TO_OBJECT

# Map of object type to channel and id used within that channel, used for grid and gridworld representations
# Agent and Start are considered equivalent
OBJECT_TO_CHANNEL_AND_IDX = {
    'empty'         : (0, 0),
    'wall'          : (0, 1),
    'agent': (1, 1),
    'start': (1, 1),
    'goal'          : (2, 1),
}

# Map of object type to feature dimension, used for graph representations
# Agent and Start are considered equivalent
OBJECT_TO_FEATURE_DIM = {
    'empty'         : 0,
    'wall'          : 1,
    'agent'         : 2,
    'start'         : 2,
    'goal'          : 3,
}


class GridNavDatasetGenerator:

    def __init__(self, dataset_meta: Dict[str, Any], batches_meta: List[Dict[str, Any]], save_dir: str):
        self.dataset_meta = dataset_meta
        self.batches_meta = batches_meta
        self.base_dir = str(Path(__file__).resolve().parent) + '/datasets/'
        self.save_dir = self.base_dir + save_dir + '/'
        self.generated_batches = []
        self.generated_labels = []
        self.generated_label_contents = []
        self.data_type = self.dataset_meta['data_type']

        if self.data_type == 'grid':
            dim0, dim1 = (int((self.dataset_meta['data_dim'][0]-1)/2), int((self.dataset_meta['data_dim'][1]-1)/2))
            self.feature_shape = (dim0, dim1, 4)
        elif self.data_type == 'gridworld':
            self.feature_shape = (self.dataset_meta['data_dim'][0], self.dataset_meta['data_dim'][1], 3)
        elif self.data_type == 'graph':
            self.feature_shape = None #TODO: decide if used later
        else:
            raise KeyError(f"Data Type '{self.data_type}' not recognised")

    def generate_dataset(self, normalise_difficulty: bool = True):

        for i, batch_meta in enumerate(self.batches_meta):
            batch_g = BatchGenerator(batch_meta, self.dataset_meta)
            batch_features, batch_label_ids, batch_label_contents = batch_g.generate_batch()
            self.generated_batches.append(batch_features)
            self.generated_labels.append(batch_label_ids)
            self.generated_label_contents.append(batch_label_contents)

        if normalise_difficulty: self.normalise_difficulty()

        # creates folder if it does not exist.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i in range(len(self.generated_batches)):
            self.save_batch(self.generated_batches[i], self.generated_labels[i],
                            self.generated_label_contents[i], self.batches_meta[i])

        self.save_dataset_meta()

    def save_batch(self, batch_data: List[Any], batch_labels: np.ndarray,
                   batch_label_contents: Dict[int, Any], batch_meta: Dict[str, Any]):

        filename = self.save_dir + batch_meta['output_file']
        if isinstance(batch_labels, np.ndarray):
            batch_labels = torch.tensor(batch_labels)

        # need to save (data, labels) and (label_contents, metadata) in 2 separate files because of limitations of
        # save_graphs()
        if self.data_type=='graph':
            entry = {'label_contents': batch_label_contents, 'batch_meta': batch_meta}
            dgl.data.utils.save_graphs(filename, batch_data, {'labels': batch_labels})
            filename += '.meta'
        else:
            entry = {'data': batch_data, 'labels': batch_labels,
                     'label_contents': batch_label_contents, 'batch_meta': batch_meta}
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def save_dataset_meta(self):

        entry = self.dataset_meta

        filename = self.save_dir + self.dataset_meta['output_file']
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def normalise_difficulty(self):
        pass  # TODO: implement


class BatchGenerator:

    def __new__(cls, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):

        if batch_meta['task_structure'] == 'maze':
            instance = super().__new__(MazeBatch)
        elif batch_meta['task_structure'] == 'rooms_unstructured_layout':
            instance = super().__new__(RoomsUnstructuredBatch)
        else:
            raise KeyError("Task Structure was not recognised")

        instance.__init__(batch_meta, dataset_meta)

        return instance

    def __init__(self):
        raise RuntimeError(f"{type(self)} is a Class Factory. Assign it to a variable. ")


class Batch:

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        self.batch_meta = batch_meta
        self.dataset_meta = dataset_meta
        self.data_type = dataset_meta['data_type']

    def generate_batch(self):

        features = self.generate_data()
        if self.data_type == 'gridworld':
            pass
        elif self.data_type == 'grid':
            features = self.encode_gridworld_to_grid(features)
            #TODO: cleanup and put as unit test
            # features3 = self.encode_grid_to_gridworld(features2)
            # assert np.array_equal(features, features3)
        elif self.data_type == 'graph':
            features = self.encode_gridworld_to_graph(features)
        solutions = self.generate_solutions(features)
        label_ids, label_content = self.generate_labels(solutions)

        return features, label_ids, label_content

    def generate_data(self):
        raise NotImplementedError

    def generate_labels(self, solutions: List[List[Tuple]]) -> Tuple[
        np.ndarray, Dict[int, Any]]:
        # call a specific quantity with
        # labels[dataset_meta['label_descriptors'].index('wanted_label_descriptor')][label_id]

        batch_id = [self.batch_meta['batch_id']] * self.batch_meta['batch_size']  # from batch_meta

        task_difficulty = self.task_difficulty(solutions, self.dataset_meta['difficulty_descriptors'])

        seed = [0] * self.batch_meta['batch_size']  # TODO: implement

        # task structure one-hot vector [0,1]
        task_structure = np.zeros((self.batch_meta['batch_size'], len(self.dataset_meta['task_structure_descriptors'])),
                                  dtype=int)
        batch_task_structure_idx = self.dataset_meta['task_structure_descriptors'].index(self.batch_meta['task_structure'])
        task_structure[:, batch_task_structure_idx] = 1

        # TODO: make robust against change of label descriptors (or declare label descriptors as a class var)
        # TODO: assert checks that they are all the right dimensions
        label_ids = np.arange(self.batch_meta['batch_size'])
        label_contents = {0: task_difficulty, 1: task_structure, 2: batch_id, 3: seed, 4: solutions}

        return label_ids, label_contents

    #TODO implement
    @staticmethod
    def generate_solutions(features) -> List[List[Tuple]]:

        solutions = []
        for layout in features:
            optimal_trajectory = (0,0)
            solutions.append(optimal_trajectory)

        return solutions

    @staticmethod
    def task_difficulty(solutions: List[List[Tuple]], difficulty_descriptors: List[str]) -> np.ndarray:

        difficulty_metrics = np.zeros((len(solutions), len(difficulty_descriptors)))

        if 'shortest_path' in difficulty_descriptors:
            shortest_path_ind = difficulty_descriptors.index('shortest_path')
            shortest_path_length = [len(solutions[i]) for i in range(len(solutions))]
            difficulty_metrics[:, shortest_path_ind] = shortest_path_length

        # TODO: add other difficulty metrics

        return difficulty_metrics

    @staticmethod
    def encode_maze_to_gridworld(mazes: Union[Maze, List[Maze]]) -> np.ndarray:

        if isinstance(mazes, Maze):
            mazes = [mazes]

        # Obtain the different channels
        grids = np.array([mazes[i].grid for i in range(len(mazes))])
        start_positions_indices = np.array([[i, mazes[i].start[0], mazes[i].start[1]] for i in range(len(mazes))])
        goal_positions_indices = np.array([[i, mazes[i].end[0], mazes[i].end[1]] for i in range(len(mazes))])
        start_position_channels, goal_position_channels = (np.zeros(grids.shape) for i in range(2))
        start_position_channels[tuple(start_positions_indices.T)] = 1
        goal_position_channels[tuple(goal_positions_indices.T)] = 1

        # merge
        features = np.stack((grids, start_position_channels, goal_position_channels), axis=-1)

        return features

    @staticmethod
    def encode_minigrid_to_gridworld(envs: List[MiniGridEnv]) -> np.ndarray:
        minigrid_grid_arrays = [env.grid.encode()[:, :, 0] for env in envs]

        for i in range(len(envs)):
            minigrid_grid_arrays[i][tuple(envs[i].agent_pos)] = Minigrid_OBJECT_TO_IDX['agent']
        minigrid_grid_arrays = np.array(minigrid_grid_arrays)

        objects_idx = np.unique(minigrid_grid_arrays)
        object_instances = [Minigrid_IDX_TO_OBJECT[obj] for obj in objects_idx]

        features = np.zeros((*minigrid_grid_arrays.shape, 3))
        for obj in object_instances:
            try:
                c = OBJECT_TO_CHANNEL_AND_IDX[obj][0]
                mini_v = Minigrid_OBJECT_TO_IDX[obj]
                v = OBJECT_TO_CHANNEL_AND_IDX[obj][1]
                features[..., c] = np.where(minigrid_grid_arrays == mini_v, v, features[..., c])
            except KeyError:
                raise KeyError("Mismatch between Minigrid generated objects and admissible objects for dataset.")

        return features

    @staticmethod
    def encode_gridworld_to_maze(grids: np.ndarray) -> List[Maze]:
        # Set up maze generator
        mazes = [Maze() for i in range(grids.shape[0])]  # TODO: here add seed argument later
        for (maze, grid) in zip(mazes, grids):
            maze.grid = np.int8(grid[...,0])
            maze.start = tuple(np.argwhere(grid[..., 1] == 1)[0])
            maze.end = tuple(np.argwhere(grid[..., 2] == 1)[0])

        return mazes

    @staticmethod
    def encode_gridworld_to_minigrid(mazes: np.ndarray, config: Dict) -> List[Any]:
        raise NotImplementedError

    @staticmethod
    def encode_gridworld_to_grid(gridworlds: np.ndarray):
        # gridworls shape: [m, odd, odd, 3]
        assert gridworlds.shape[1] % 2 == 1 and gridworlds.shape[2] % 2 == 1, \
            "Inputted Gridworlds do not have a layout of odd dimensions"
        assert gridworlds.shape[-1] == 3, "Inputted Gridworlds do not have 3 channels"
        grid_layout_dim = (gridworlds.shape[0], int(np.floor(gridworlds.shape[1]/2)), int(np.floor(gridworlds.shape[2]/2)), 2)
        grid_layouts = np.zeros(grid_layout_dim)

        layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]
        empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grid_layouts.shape[0]):
            for i in range(grid_layouts.shape[1]):
                for j in range(grid_layouts.shape[2]):
                    ind_gridworld = (m, int(i*2+1), int(j*2+1), layout_channel)
                    ind_gridworld_right = list(ind_gridworld)
                    ind_gridworld_right[2] += 1
                    ind_gridworld_bot = list(ind_gridworld)
                    ind_gridworld_bot[1] += 1
                    if gridworlds[ind_gridworld] == empty_idx:
                        if gridworlds[tuple(ind_gridworld_right)] == empty_idx:
                            grid_layouts[m,i,j,0] = 1
                        if gridworlds[tuple(ind_gridworld_bot)] == empty_idx:
                            grid_layouts[m, i, j, 1] = 1

        #TODO: use object dictionary
        start_channels, goal_channels = (np.zeros(grid_layout_dim[:-1]) for i in range(2))
        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        start_inds_grid = (start_inds_gridworld[0], np.floor(start_inds_gridworld[1] / 2).astype(int),
                           np.floor(start_inds_gridworld[2] / 2).astype(int))
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]]
                                       == OBJECT_TO_CHANNEL_AND_IDX['goal'][1])
        goal_inds_grid = (goal_inds_gridworld[0], np.floor(goal_inds_gridworld[1] / 2).astype(int),
                           np.floor(goal_inds_gridworld[2] / 2).astype(int))

        #TODO use object dictionary
        start_channels[start_inds_grid] = 1
        goal_channels[goal_inds_grid] = 1

        # merge
        grids = np.stack((grid_layouts[...,0], grid_layouts[...,1], start_channels, goal_channels), axis=-1)
        return grids

    @staticmethod
    def encode_grid_to_gridworld(grids: Union[np.ndarray, torch.Tensor], layout_only=False):

        if layout_only: expected_channels = 2
        else: expected_channels = 4

        #TODO: perf remodel to handle GPU
        tensor = False
        if torch.is_tensor(grids):
            tensor = True
            device = grids.device
            assert len(grids.shape) == 4, f"Grids Tensor has {len(grids.shape)} dimensions. Expected {4}"
            if grids.shape[-1] != expected_channels and grids.shape[1] == expected_channels:
                grids = torch.permute(grids, (0, 2, 3, 1)) # (B, C, H, W) -> (B, H, W, C)
            grids = grids.detach().cpu().numpy()

        assert grids.shape[-1] == expected_channels, f"Inputted Grids have {grids.shape[-1]} channels. Expected {expected_channels}"

        gridworlds_layout_dim = (
        grids.shape[0], int(2 * grids.shape[1] + 1), int(2 * grids.shape[2] + 1))
        gridworlds_layouts = np.ones(gridworlds_layout_dim) * OBJECT_TO_CHANNEL_AND_IDX['wall'][-1]

        #TODO: add grid dictionary descriptions
        gridworlds_layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0] #could be used with a "sort"
        gridworlds_empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grids.shape[0]):
            for i in range(grids.shape[1]):
                for j in range(grids.shape[2]):
                    if grids[m,i,j,0] == 1 or grids[m,i,j,1] == 1:
                        i_gridworld, j_gridworld = (2 * i + 1, 2 * j + 1)
                        gridworlds_layouts[m,i_gridworld,j_gridworld] = gridworlds_empty_idx
                        if grids[m,i,j,0] == 1:
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 1] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 2] = gridworlds_empty_idx
                        if grids[m,i,j,1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld + 2, j_gridworld] = gridworlds_empty_idx
                        # clique rule
                        if grids[m,i,j,0] == grids[m,i,j,1] == grids[m,i+1,j,0] == grids[m,i,j+1,1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld + 1] = gridworlds_empty_idx

        if layout_only:
            gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape,1))
        #TODO: use object dictionary
        else:
            start_channels, goal_channels = (np.zeros(gridworlds_layout_dim) for i in range(2))

            start_inds_grids = np.where(grids[..., 2] == 1)
            start_inds_gridworlds = (start_inds_grids[0], (2*start_inds_grids[1] + 1).astype(int),
                               (2*start_inds_grids[2] + 1).astype(int))
            goal_inds_grids = np.where(grids[..., 3] == 1)
            goal_inds_gridworlds = (goal_inds_grids[0], (2*goal_inds_grids[1] + 1).astype(int),
                               (2*goal_inds_grids[2] + 1).astype(int))

            start_channels[start_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['start'][1]
            goal_channels[goal_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['goal'][1]

            # merge
            gridworlds = np.stack((gridworlds_layouts, start_channels, goal_channels), axis=-1)

        if tensor:
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)

        return gridworlds

    @staticmethod
    def encode_gridworld_to_graph(gridworlds: np.ndarray):
        # Graph feature shape [empty, wall, start, goal]
        # Graph nodes: ((gw_dim - 1)/2)**2

        # gridworlds shape: [m, odd, odd, 3]
        assert gridworlds.shape[1] % 2 == 1 and gridworlds.shape[2] % 2 == 1, \
            "Inputted Gridworlds do not have a layout of odd dimensions"
        assert gridworlds.shape[-1] == 3, "Inputted Gridworlds do not have 3 channels"

        dim_grid = (int((gridworlds.shape[1] - 1) / 2), int((gridworlds.shape[2] - 1) / 2))

        layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]
        empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        adj = Batch.encode_gridworld_layout_to_adj(gridworlds[...,layout_channel], empty_idx) # M, N, N

        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]]
                                       == OBJECT_TO_CHANNEL_AND_IDX['goal'][1])

        start_nodes_graph = (start_inds_gridworld[0],
                             ((start_inds_gridworld[1] - 1)/2 * dim_grid[0] + (start_inds_gridworld[2] - 1)/2).astype(int))
        goal_nodes_graph = (goal_inds_gridworld[0],
                             ((goal_inds_gridworld[1] - 1)/2 * dim_grid[0] + (goal_inds_gridworld[2] - 1)/2).astype(int))
        active_nodes_graph = np.where(adj.sum(axis=1)!=0)
        wall_nodes_graph = np.where(adj.sum(axis=1)==0)


        feats = np.zeros((adj.shape[0], adj.shape[1], 4)) # M, N, D
        feats[(*start_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['start']]*len(start_nodes_graph[0])))] = 1
        feats[(*goal_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['goal']]*len(goal_nodes_graph[0])))] = 1
        if wall_nodes_graph[0].size != 0: #only if array not empty.
            feats[(*wall_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['wall']]*len(wall_nodes_graph[0])))] = 1
        # empty features are the active nodes, removing the nodes having goal or start feature
        feats[(*active_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']]*len(active_nodes_graph[0])))] = 1
        feats[(*goal_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']] * len(goal_nodes_graph[0])))] = 0
        feats[(*start_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']] * len(start_nodes_graph[0])))] = 0
        feats = torch.tensor(feats)
        # check all features are one-hot.
        assert (feats.sum(axis=-1) == 1).all()

        graphs = []
        for m in range(adj.shape[0]):
            src, dst = np.nonzero(adj[m])
            g = dgl.graph((src, dst), num_nodes=len(feats[m]))
            g.ndata['feat'] = feats[m]
            graphs.append(g)

        return graphs

    #Note: Returns the gridworld in one given permutation
    @staticmethod
    def encode_graph_to_gridworld(graphs: Union[List[dgl.DGLGraph], tuple],
                                  attributes:Tuple[str]=("empty", "wall", "start", "goal"),
                                  used_attributes:Tuple[str]=("empty", "start", "goal" ),
                                  output_dtype: str = 'tensor'):

        def get_gw_inds(nodes_tuple:Tuple[np.ndarray], n_nodes, mapping=lambda x : 2*x+1):
            inds_tuple = []
            for nodes in nodes_tuple:
                inds = (nodes[1] // np.sqrt(n_nodes), nodes[1] % np.sqrt(n_nodes))
                inds = tuple([mapping(i.astype(int)) for i in inds])
                inds = (nodes[0],) + inds
                inds_tuple.append(inds)
            return tuple(inds_tuple)

        #Note: modes 2 and 3 can only work for layouts with 1-to-1 cell-node characterisation
        possible_modes = {
            ():0, ("",):0,                              #0: Layout only from A
            ("start", "goal"): 1,                       #1: Layout from A, start and goal from Fx
            ("empty", "start", "goal"): 2,              #2: Layout, start, goal from Fx
            ("empty", "wall", "start", "goal"): 3,      #3: Layout, start, goal from Fx, may form impossible layouts
        }

        if output_dtype == 'tensor':
            device = graphs[0].device

        try:
            mode = possible_modes[used_attributes]
        except KeyError:
            raise AttributeError(f"Gridworld encoding from {used_attributes} is not possible.")

        if isinstance(graphs, tuple):
            A, fx = graphs
            n_nodes = fx.shape[-2]
            feat_dim = fx.shape[-1]
            A = torch.reshape(A, (A.shape[0], -1, 2))
        elif isinstance(graphs[0], dgl.DGLGraph):
            n_nodes = graphs[0].num_nodes() #assumption that all graphs have same number of nodes
            feat_dim = graphs[0].ndata['feat'].shape
            assert n_nodes % np.sqrt(n_nodes) == 0 # we are assuming square layout

            A = np.empty((len(graphs), n_nodes, n_nodes))
            fx = np.empty((len(graphs), *feat_dim))
            for m in range(len(graphs)):
                A[m] = graphs[m].adj().cpu().to_dense().numpy()
                fx[m] = graphs[m].ndata['feat'].cpu().numpy()
            A = Batch.encode_adj_to_reduced_adj(A)
        else:
            raise RuntimeError(f"data format {type(graphs)} is not supported by function. Format supported are"
                                 f"List[dgl.DGLGraph], tuple[tensor, tensor]")

        gridworld_layout_dim = (int(2 * np.sqrt(n_nodes) + 1), int(2 * np.sqrt(n_nodes) + 1))

        # Modes for which we need A
        if mode in [0, 1]:
            gridworlds_layouts = Batch.encode_reduced_adj_to_gridworld_layout(A, gridworld_layout_dim)
            if mode in [0, ]:
                gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape, 1))
        # Modes for which we need Fx[start, goal]
        if mode in [1, 2,]: #[1,2,3] when implemented
            gridworlds = np.zeros((fx.shape[0], *gridworld_layout_dim, 3))

            start_nodes = np.where(fx[..., attributes.index('start')] == 1)
            goal_nodes = np.where(fx[..., attributes.index('goal')] == 1)
            # start_inds = (start_nodes[1] // np.sqrt(n_nodes), start_nodes[1] % np.sqrt(n_nodes))
            # goal_inds = (goal_nodes[1] // np.sqrt(n_nodes), goal_nodes[1] % np.sqrt(n_nodes))
            # start_inds, goal_inds = (tuple([2 * i.astype(int) + 1 for i in tup]) for tup in (start_inds, goal_inds))
            # start_inds = (start_nodes[0],) + start_inds
            # goal_inds = (goal_nodes[0],) + goal_inds

            start_inds, goal_inds = get_gw_inds((start_nodes, goal_nodes), n_nodes)

            gridworlds[(*start_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['start'][0]] * gridworlds.shape[0]))] = \
                OBJECT_TO_CHANNEL_AND_IDX['start'][1]
            gridworlds[(*goal_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['goal'][0]] * gridworlds.shape[0]))] = \
                OBJECT_TO_CHANNEL_AND_IDX['goal'][1]
            if mode in [1,]: #add layout from adjacency
                gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['wall'][0]] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1] * \
                                                                    gridworlds_layouts
            elif mode in [2,]: #add layout from empty nodes
                # set all cells to wall in layout channel
                gridworlds[..., np.array([OBJECT_TO_CHANNEL_AND_IDX['wall'][0]])] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1]

                # set all non wall (empty, goal, start) to empty
                #start, goal
                gridworlds[(*start_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
                gridworlds[(*goal_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]

                #empty
                empty_nodes = np.where(fx[..., attributes.index('empty')] == 1)
                # empty_inds = (empty_nodes[1] // np.sqrt(n_nodes), empty_nodes[1] % np.sqrt(n_nodes))
                # empty_inds = tuple([2 * i.astype(int) + 1 for i in empty_inds])
                # empty_inds = (empty_nodes[0],) + goal_inds
                empty_inds, = get_gw_inds((empty_nodes,), n_nodes)

                gridworlds[(*empty_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * empty_inds[0].shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
        elif mode in [3,]:
            raise NotImplementedError(f"Gridworld encoding from {used_attributes} not yet implemented.")

        if output_dtype == 'tensor':
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)

        return gridworlds

    @staticmethod
    def encode_reduced_adj_to_gridworld_layout(A: Union[np.ndarray, torch.tensor], layout_dim):

        n_nodes = A.shape[1] + 1
        gridworlds_layout_dim = (A.shape[0], *layout_dim)
        assert gridworlds_layout_dim == (A.shape[0], int(2 * np.sqrt(n_nodes) + 1), int(2 * np.sqrt(n_nodes) + 1))
        gridworlds_layouts = np.ones(gridworlds_layout_dim) * OBJECT_TO_CHANNEL_AND_IDX['wall'][-1]
        gridworlds_empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]


        for m in range(A.shape[0]):
            for n in range(A[m].shape[0]):
                # horizontal edge:
                if (A[m,n,:] == 1).any():
                    i_n = n // int(np.sqrt(n_nodes))
                    j_n = n % int(np.sqrt(n_nodes))
                    i, j = 2 * i_n + 1, 2 * j_n + 1
                    gridworlds_layouts[m, i, j] = gridworlds_empty_idx
                    # TODO: is this cheating?
                    if A[m,n,0] == 1 and (j+2)<gridworlds_layouts.shape[2]: #row edge is present
                        gridworlds_layouts[m, i, j + 1] = gridworlds_empty_idx
                        gridworlds_layouts[m, i, j + 2] = gridworlds_empty_idx
                    if A[m,n,1] == 1 and (i+2)<gridworlds_layouts.shape[1]: #col edge is present
                        gridworlds_layouts[m, i + 1, j] = gridworlds_empty_idx
                        gridworlds_layouts[m, i + 2, j] = gridworlds_empty_idx
                    # clique rule
                    if i + 1 < gridworlds_layouts.shape[1] and j + 1 < gridworlds_layouts.shape[2]:
                        if n+int(np.sqrt(n_nodes)) < A[m].shape[0]:
                            if A[m,n,0] == A[m,n,1] == A[m,n+1,1] == A[m,n+int(np.sqrt(n_nodes)),0] == 1:
                                gridworlds_layouts[m, i + 1, j + 1] = gridworlds_empty_idx

        if torch.is_tensor(A):
            gridworlds_layouts = torch.tensor(gridworlds_layouts).to(A.device)
        return gridworlds_layouts

    @staticmethod
    def encode_gridworld_layout_to_adj(layouts: np.ndarray, empty_idx=0):
        #layouts shape: [m, odd, odd]
        assert layouts.shape[1] % 2 == 1 and layouts.shape[2] % 2 == 1, \
            "Inputted Gridworlds Layouts do not have odd number of elements"
        assert len(layouts.shape) == 3, "Layout not inputted correctly. Input layouts as (m, row, col)"
        node_inds_i, node_inds_j = [i for i in range(1, layouts.shape[1], 2)], [i for i in range(1, layouts.shape[2], 2)]
        A = np.zeros((layouts.shape[0], len(node_inds_i) * len(node_inds_j), len(node_inds_i) * len(node_inds_j)))

        for m in range(A.shape[0]):
            for i_A, i_gw in enumerate(node_inds_i):
                for j_A, j_gw in enumerate(node_inds_j):
                    if layouts[m, i_gw, j_gw] == empty_idx:
                        ind_gw_right = (m, i_gw, j_gw + 1)
                        ind_gw_bot = (m, i_gw + 1, j_gw)
                        if layouts[ind_gw_right] == empty_idx:
                            ind_A_right = (m, i_A*len(node_inds_i)+j_A, i_A*len(node_inds_i)+j_A+1)
                            A[ind_A_right] = 1
                        if layouts[ind_gw_bot] == empty_idx:
                            ind_A_bot = (m, i_A*len(node_inds_i)+j_A, (i_A+1)*len(node_inds_i)+j_A)
                            A[ind_A_bot] = 1
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A

    @staticmethod
    def encode_adj_to_reduced_adj(adj: Union[np.ndarray, torch.tensor]):
        # the last sqrt(n)-1 col edges will always be 0.
        #adj shape (m, n, n)
        if torch.is_tensor(adj):
            A = torch.zeros((adj.shape[0], adj.shape[1] - 1, 2)).to(adj.device)
        elif isinstance(adj, np.ndarray):
            A = np.zeros((adj.shape[0], adj.shape[1] - 1, 2))
        dim_grid = int(np.sqrt(adj.shape[1])) #only for square grids
        A[...,0] = adj.diagonal(1,1,2) # row edges
        A[:,:-dim_grid+1,1] = adj.diagonal(dim_grid, 1, 2) # col edges

        return A # (m, n-1, 2)

    @staticmethod
    def encode_reduced_adj_to_adj(adj_r: np.ndarray):
        # only for square grids
        #adj shape (m, n-1, 2)
        A = np.empty((adj_r.shape[0], adj_r.shape[1] + 1, adj_r.shape[1] + 1))
        dim_grid = int(np.sqrt(A.shape[1]))
        for m in range(A.shape[0]):
            A[m] = np.diag(adj_r[m,:,0], k = 1)
            A[m] += np.diag(adj_r[m,:-dim_grid+1,1], k = dim_grid)
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A # (m, n, n)

    @staticmethod
    def augment_adj(n: int, transforms: torch.tensor):
        # transforms represent all allowable permutations in a 2D grid
        nodes_inds = torch.arange(0, n, dtype=torch.int) # node indices in adjacency matrix
        i_n, j_n = nodes_inds.div(int(n**.5), rounding_mode='floor'), nodes_inds % int(n**.5) #corresponding indices in grid space, top left corner origin
        ij_n = torch.stack([i_n, j_n], dim=0) # D N
        ij_n = einops.repeat(ij_n, 'd n ->  p d n', p=transforms.shape[0]) # P D N
        c = torch.tensor([int((n ** .5 - 1) / 2), int((n ** .5 - 1) / 2)], dtype=torch.int).unsqueeze(1) # D 1
        c = einops.repeat(c, 'd n -> p d n', p=transforms.shape[0]) # P D 1
        ij_c = (ij_n - c) # P D=2 N. Corresponding indices with origin in the middle of the grid
        # coordinate transform 1 done: origin in grid space
        ij_t = torch.matmul(transforms, ij_c) # P 2 2 @ P D=2 N -> P D=2 N # rotate the coordinate axis
        # coordinate transform 2 done: axis rotated
        ij_f = (ij_t + c) # add the centroid to come back to a top left corner coordinate system
        ij2n = torch.tensor([int(n**0.5), 1], dtype=torch.int).unsqueeze(1) # D 1 #transformation matrix to get node ordering, left to right, top to bottom in graph space
        ij2n = einops.repeat(ij2n, 'd n -> p n d', p=transforms.shape[0]) # P 1 D
        nodes_inds_t = torch.matmul(ij2n, ij_f).squeeze() # P 1 D @ P D N = P D N #recover the transformed indices
        return nodes_inds_t


class MazeBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        super().__init__(batch_meta, dataset_meta)

    def generate_data(self):
        # Set up maze generator
        maze_generator = Maze()  # TODO: here add seed argument later
        maze_size_arg = [int((x - 1) / 2) for x in self.dataset_meta['data_dim']]

        # Set up generating algorithm
        if self.batch_meta['generating_algorithm'] == 'Prims':
            maze_generator.generator = Prims(*maze_size_arg)
        else:
            raise KeyError(f"Maze generating algorithm '{self.batch_meta['generating_algorithm']}' was not recognised")

        batch_features = []
        for i in range(self.batch_meta['batch_size']):
            # maze_generator.set_seed(self.batch_seeds[i]) #TODO
            maze_generator.generate()
            maze_generator.generate_entrances(False, False)
            features = self.encode_maze_to_gridworld(maze_generator)
            batch_features.append(features)

        batch_features = np.squeeze(batch_features)
        return batch_features

    # def generate_solutions(self, features) -> List[List[Tuple]]:
    #
    #     solutions = []
    #     # Set up solving algorithm
    #     if self.batch_meta['solving_algorithm'] == 'ShortestPaths':
    #         maze_generator.solver = ShortestPaths()
    #     else:
    #         raise KeyError("Maze solving algorithm was not recognised")
    #
    #     for i in range(self.batch_meta['batch_size']):
    #         maze_generator.solve()
    #         optimal_trajectory = maze_generator.solutions[0]  # TODO Check this is the right one.
    #         solutions.append(optimal_trajectory)


class RoomsUnstructuredBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        super().__init__(batch_meta, dataset_meta)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, Any]]:
        # Set up generator
        #TODO do seed properly
        envs = [MultiRoomEnv(minNumRooms=4, maxNumRooms=12, minRoomSize=5, maxRoomSize=9,
                             grid_size=self.dataset_meta['data_dim'][0], odd=True,
                             seed=np.random.randint(1e6)) for i in range(self.batch_meta['batch_size'])]

        batch_features = self.encode_minigrid_to_gridworld(envs)

        return batch_features

    #TODO: Fix
    # def generate_solutions(self, features) -> List[List[Tuple]]:
    #
    #     solutions = []
    #     for layout in features:
    #         optimal_trajectory = (0,0)  # TODO Check this is the right one.
    #         solutions.append(optimal_trajectory)
    #
    #     return solutions


if __name__ == '__main__':
    dataset_meta = {
        'output_file': 'dataset.meta',
        'data_type': 'graph', #types: gridworld, grid, graph
        'data_dim': (27, 27),  # TODO: assert odd. Note: always in "gridworld" type
        'task_type': 'find_goal',
        'label_descriptors': [
            'difficulty_metrics',
            'task structure',
            'batch_id',
            'seed',
            'optimal_trajectory',
        ],
        'difficulty_descriptors': [
            'shortest_path',
            'full_exploration',
        ],
        'task_structure_descriptors': [
            'rooms_unstructured_layout',
            'rooms_square_layout',
            'maze',
            'dungeon',
        ],
        'feature_descriptors': [
            'walls',
            'start_position',
            'goal_position',
        ],
        'generating_algorithms_descriptors': [
            'Prims',
        ],
        'solving_algorithms_descriptors': [
            'ShortestPaths',
        ],
    }

    # batches_meta = [
    #     {
    #         'output_file': 'batch_0.data',
    #         'batch_size': 100,
    #         'batch_id': 0,
    #         'task_structure': 'rooms_unstructured_layout',
    #         'generating_algorithm': 'Prims',
    #         'generating_algorithm_options': [
    #
    #         ],
    #         'solving_algorithm': 'ShortestPaths',
    #         'solving_algorithm_options': [
    #
    #         ],
    #     },
    # ]

    batches_meta = [
        {
            'output_file': 'batch_0.data',
            'batch_size': 200,
            'batch_id': 0,
            'task_structure': 'rooms_unstructured_layout',
            'generating_algorithm': 'Minigrid_MultiRoom',
            'generating_algorithm_options': [

            ],
            'solving_algorithm': 'ShortestPaths',
            'solving_algorithm_options': [

            ],
        },
        # {
        #     'output_file': 'batch_1.data',
        #     'batch_size': 1000,
        #     'batch_id': 1,
        #     'task_structure': 'rooms_unstructured_layout',
        #     'generating_algorithm': 'Minigrid_MultiRoom',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_2.data',
        #     'batch_size': 1000,
        #     'batch_id': 2,
        #     'task_structure': 'rooms_unstructured_layout',
        #     'generating_algorithm': 'Minigrid_MultiRoom',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_3.data',
        #     'batch_size': 1000,
        #     'batch_id': 3,
        #     'task_structure': 'rooms_unstructured_layout',
        #     'generating_algorithm': 'Minigrid_MultiRoom',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_4.data',
        #     'batch_size': 1000,
        #     'batch_id': 4,
        #     'task_structure': 'rooms_unstructured_layout',
        #     'generating_algorithm': 'Minigrid_MultiRoom',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        {
            'output_file': 'batch_5.data',
            'batch_size': 200,
            'batch_id': 5,
            'task_structure': 'maze',
            'generating_algorithm': 'Prims',
            'generating_algorithm_options': [

            ],
            'solving_algorithm': 'ShortestPaths',
            'solving_algorithm_options': [

            ],
        },
        # {
        #     'output_file': 'batch_6.data',
        #     'batch_size': 1000,
        #     'batch_id': 6,
        #     'task_structure': 'maze',
        #     'generating_algorithm': 'Prims',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_7.data',
        #     'batch_size': 1000,
        #     'batch_id': 7,
        #     'task_structure': 'maze',
        #     'generating_algorithm': 'Prims',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_8.data',
        #     'batch_size': 1000,
        #     'batch_id': 8,
        #     'task_structure': 'maze',
        #     'generating_algorithm': 'Prims',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        # {
        #     'output_file': 'batch_9.data',
        #     'batch_size': 1000,
        #     'batch_id': 9,
        #     'task_structure': 'maze',
        #     'generating_algorithm': 'Prims',
        #     'generating_algorithm_options': [
        #
        #     ],
        #     'solving_algorithm': 'ShortestPaths',
        #     'solving_algorithm_options': [
        #
        #     ],
        # },
        {
            'output_file': 'test_batch_0.data',
            'batch_size': 200,
            'batch_id': 10,
            'task_structure': 'rooms_unstructured_layout',
            'generating_algorithm': 'Minigrid_MultiRoom',
            'generating_algorithm_options': [

            ],
            'solving_algorithm': 'ShortestPaths',
            'solving_algorithm_options': [

            ],
        },
        {
            'output_file': 'test_batch_1.data',
            'batch_size': 200,
            'batch_id': 91,
            'task_structure': 'maze',
            'generating_algorithm': 'Prims',
            'generating_algorithm_options': [

            ],
            'solving_algorithm': 'ShortestPaths',
            'solving_algorithm_options': [

            ],
        },
    ]

    attributes = ("empty", "wall", "start", "goal")
    used_attributes = ("empty", "start", "goal")

    transforms = torch.tensor([[[1, 0], [0, 1]],
                               [[1, 0], [0, -1]],
                               [[0, 1], [1, 0]],
                               [[0, 1], [-1, 0]],
                               [[-1, 0], [0, 1]],
                               [[-1, 0], [0, -1]],
                               [[0, -1], [1, 0]],
                               [[0, -1], [-1, 0]]], dtype=torch.int)
    grid = np.zeros((1, 3, 3, 4))
    grid[0, 0, 1, 0] = 1
    grid[0, 1, 0, 0] = 1
    grid[0, 1, 1, 0] = 1
    grid[0, 2, 1, 0] = 1
    grid[0, 0, 0, 1] = 1
    grid[0, 0, 1, 1] = 1
    grid[0, 1, 1, 1] = 1
    grid[0, 1, 2, 1] = 1

    gridworld = Batch.encode_grid_to_gridworld(grid)
    gridworld[0, 1, 3, 1] = 1
    gridworld[0, 3, 5, 2] = 1
    gridworld = np.tile(gridworld, (10, 1, 1, 1))

    graphs = Batch.encode_gridworld_to_graph(gridworld)
    rec_gridworlds = Batch.encode_graph_to_gridworld(graphs, attributes=attributes, used_attributes=used_attributes, output_dtype='dewd')
    assert (rec_gridworlds == gridworld).all()
    A = []
    for g in graphs:
        A.append(g.adj().cpu().to_dense().numpy())
    A = torch.tensor(A) # M x N x N
    num_nodes = A.shape[-1]
    permutations = Batch.augment_adj(num_nodes, transforms)

    permuted_A = [A[:,permutations[i].long()][:,:, permutations[i].long()] for i in range(permutations.shape[0])]
    permuted_A = torch.stack(permuted_A, dim=1) # Mx P x Nx N
    permuted_A_f = permuted_A.reshape(permuted_A.shape[0]*permuted_A.shape[1],*permuted_A.shape[2:])


    reduced_adj = Batch.encode_adj_to_reduced_adj(permuted_A_f)
    gridworld_layouts = Batch.encode_reduced_adj_to_gridworld_layout(reduced_adj, layout_dim = gridworld.shape[1:3])
    gridworld_layouts = gridworld_layouts.reshape(10, 8, 7, 7)

    for i in range(gridworld_layouts.shape[1]):
        for j in range(gridworld_layouts.shape[1]):
            if i != j:
                assert not (gridworld_layouts[0][i] == gridworld_layouts[0][j]).all()

    task_structures = []
    dataset_size = 0
    for batch_meta in batches_meta:
        if batch_meta['task_structure'] not in task_structures:
            task_structures.append(batch_meta['task_structure'])
        dataset_size += batch_meta['batch_size']
    task_structures = '-'.join(task_structures)
    dataset_directory = f"ts={task_structures}-x={dataset_meta['data_type']}-s={dataset_size}-d={dataset_meta['data_dim'][0]}"
    dataset_directory = 'test'
    MazeGenerator = GridNavDatasetGenerator(dataset_meta=dataset_meta, batches_meta=batches_meta, save_dir=dataset_directory)
    MazeGenerator.generate_dataset()

    print("Done")
