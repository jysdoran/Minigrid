import logging

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple, Dict, List

import numpy as np
import torch
import pytorch_lightning as pl
import dgl

from torch.utils.data import random_split
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from dgl.dataloading import GraphDataLoader

#import memory_profiler
import maze_representations.util.util as util
import maze_representations.util.transforms as tr
from util import DotDict

logger = logging.getLogger(__name__)

class GridNavDataset(VisionDataset):
    """`MAZE`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``maze_levels`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    base_folder = ""
    meta = {
        "filename": "dataset.meta",
        "md5": "ceb1fb9aaface2c9669a3914ecf7d30a",
        #"label_descriptors": "label", # TODO: figure out the structure of "label"
        "data_dim": "data_dim",
        "task:": "task_type",
    }

    #@memory_profiler.profile
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        no_images = False,
        held_out_tasks: Optional[List[str]] = None,
        ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.no_images = no_images

        self._load_meta()
        dataset_path = os.path.join(self.root, self.base_folder)
        files = os.listdir(dataset_path)
        self.train_list = [f for f in files if f.endswith('.data') and not f.startswith('test')]
        self.test_list = [f for f in files if f.endswith('.data') and f.startswith('test')]

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            pickled_data = self.train_list
        else:
            pickled_data = self.test_list

        self.data: Any = []
        self.targets = []
        self.target_contents: Dict = {}
        self.batches_metadata: Any = []
        self.held_out_tasks = held_out_tasks

        if self.held_out_tasks is not None:
            logger.info("Held out tasks: {}".format(self.held_out_tasks))

        # now load the picked numpy arrays
        for file_name in pickled_data:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            self._load_data(file_path)

        if not self.data:
            raise FileNotFoundError("Dataset not found at specified location.")

    #@memory_profiler.profile
    def _load_data(self, file_path: str) -> None:

        try:
            meta_file_path = file_path + '.meta'
            extra_data_file_path = file_path + '.dgl.extra'
            with open(meta_file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                if isinstance(entry['label_contents'], DotDict):
                    entry['label_contents'] = entry['label_contents'].to_dict()
                if isinstance(entry['batch_meta'], DotDict):
                    entry['batch_meta'] = entry['batch_meta'].to_dict()

                ts = entry['label_contents']['task_structure'][0]
                assert all(ts == i for i in entry['label_contents']['task_structure']), \
                    f"Not all tasks are the same in batch {file_path}."
                if self.held_out_tasks is not None and ts in self.held_out_tasks:
                    return
                if self.data_type == 'graph':
                    if not os.path.exists(file_path): raise FileNotFoundError()
                    graphs, labels = dgl.load_graphs(file_path)
                    if os.path.exists(extra_data_file_path):
                        extra_data = dgl.load_graphs(extra_data_file_path)
                        extra_data = util.assemble_extra_data(extra_data)
                        entry['label_contents']['edge_graphs'] = extra_data['edge_graphs']

                    self.data.extend(graphs)
                    self.targets.extend(labels['labels'])
                if self.no_images:
                    try:
                        del entry["label_contents"]["images"]
                    except KeyError as e:
                        pass
                # note: will end loop prematurely if exeption is thrown, not sure this is the best pattern
                # as behavior is order dependent
                try:
                    self.batches_metadata.append(entry["batch_meta"])
                    try:
                        for key in entry["label_contents"].keys():
                            if key == '__dict__': #handles DotDict objects
                                continue
                            if isinstance(entry["label_contents"][key], list):
                                if key not in self.target_contents:
                                    self.target_contents[key] = entry["label_contents"][key]
                                else:
                                    self.target_contents[key].extend(entry["label_contents"][key])
                            elif isinstance(entry["label_contents"][key], torch.Tensor):
                                if key not in self.target_contents:
                                    self.target_contents[key] = entry["label_contents"][key]
                                else:
                                    self.target_contents[key] = \
                                        torch.cat((self.target_contents[key], entry["label_contents"][key]))
                            elif isinstance(entry["label_contents"][key], np.ndarray):
                                val = torch.from_numpy(entry["label_contents"][key])
                                if key not in self.target_contents:
                                    self.target_contents[key] = val
                                else:
                                    self.target_contents[key] = torch.cat((self.target_contents[key], val))
                            elif isinstance(entry["label_contents"][key], dict):
                                if key not in self.target_contents:
                                    self.target_contents[key] = entry["label_contents"][key]
                                else:
                                    for k in entry["label_contents"][key].keys():
                                        if k not in self.target_contents[key]:
                                            self.target_contents[key][k] = entry["label_contents"][key][k]
                                        else:
                                            self.target_contents[key][k].extend(entry["label_contents"][key][k])
                            else:
                                raise ValueError("Unsupported type for target_contents")
                    except KeyError as e:
                        raise KeyError(
                            f"{e} not found in {self.target_contents.keys()}. Mismatch in label contents "
                            f"across batch files")

                    # these ones will usually be skipped
                    self.data.extend(entry["data"])
                    self.targets.extend(entry["labels"])
                except KeyError as e:
                    pass
        except FileNotFoundError as e:
            pass

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        # if not check_integrity(path, self.meta["md5"]):
        #     raise RuntimeError("Dataset metadata file not found or corrupted.")
        with open(path, "rb") as infile:
            self.dataset_metadata = pickle.load(infile, encoding="latin1")
            self.label_descriptors = self.dataset_metadata.config.label_descriptors
            self.data_type = self.dataset_metadata.config.data_type
        self.label_to_idx = {_class: i for i, _class in enumerate(self.label_descriptors)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image: Comment - not needed unless we do image transforms
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img).type(torch.float)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return True #TODO

        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def pickle(self, path, label_id, format='bitmap', level_info=None):

        if format == 'bitmap':
            pass
        else:
            raise NotImplementedError

        if level_info is None:
            level_info = self.dataset_metadata['level_info']

        graph = self.data[label_id]
        task_structure = self.target_contents['task_structure'][label_id]
        seed = self.target_contents['seed'][label_id]

        bitmaps, start_pos, goal_pos = tr.Nav2DTransforms.graphs_to_bitmap([graph], level_info=level_info)
        bitmaps = bitmaps[0]
        start_pos = start_pos[0]
        goal_pos = goal_pos[0]

        data = {
            'task_structure': task_structure,
            'bit_map': bitmaps,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'seed': seed
            }

        with open(path, 'wb') as f:
            pickle.dump(data, f)


class CaveEscapeDataset(GridNavDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        no_images = False,
        held_out_tasks: Optional[List[str]] = None,
        ) -> None:

        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         no_images=no_images, held_out_tasks=held_out_tasks)

    def pickle(self, path, label_id, format='level_embedding', level_info=None):

        if format == 'level_embedding':
            pass
        else:
            raise NotImplementedError

        if level_info is None:
            level_info = self.dataset_metadata['level_info']
        task_structure = self.target_contents['task_structure'][label_id]
        seed = self.target_contents['seed'][label_id]

        graph = self.data[label_id]
        encoding = tr.Nav2DTransforms.dense_graph_to_minigrid([graph], level_info=level_info)[0]
        data = {
            'task_structure': task_structure,
            'encoding': encoding,
            'seed': seed,
            'level_info': level_info
            }

        with open(path, 'wb') as f:
            pickle.dump(data, f)


class CustomDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "", batch_size: int = 32, num_samples: int = 2048,
                 transform=None, num_workers: int = 0, val_data: str = 'train', no_images=True,
                 held_out_tasks:List[str]=None,
                 **kwargs):
        super().__init__()
        #sampler = dgl.dataloading.GraphDataLoader()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.transform = transform
        self.samples = {}
        self.dataset = None
        self.target_contents = None
        self.num_workers = num_workers
        self.val_data = val_data # from train or test set
        self.test = None
        self.no_images = no_images
        if held_out_tasks is None or len(held_out_tasks) == 0:
            self.held_out_tasks = None
        else:
            self.held_out_tasks = held_out_tasks #only used for train set
        self.dataset_metadata = None

        logger.info("Initializing Gridworld Navigation DataModule")
        if not self.no_images:
            logger.warning("no_images set to False, may cause memory issues.")

    def setup(self, stage=None):
        dataset_train = None
        dataset_test = None
        if stage == 'fit' or stage is None:
            dataset_train = self.dataset_object(self.data_dir, train=True, transform=self.transform,
                                            no_images=self.no_images, held_out_tasks=self.held_out_tasks)
            self.dataset_metadata = dataset_train.dataset_metadata
            if self.val_data == 'train':
                split_size = [int(0.9 * len(dataset_train)), len(dataset_train) - int(0.9 * len(dataset_train))]
                train, val = random_split(dataset_train, split_size)
            elif self.val_data == 'test':
                train = dataset_train
                dataset_test = self.dataset_object(self.data_dir, train=False, transform=self.transform)
                split_size = [int(0.5 * len(dataset_test)), len(dataset_test) - int(0.5 * len(dataset_test))]
                val, test = random_split(dataset_test, split_size)
                self.test = test
            else:
                raise ValueError(f"Incorrect val_data value: {self.val_data}. Must be either 'train' or 'test'")
            self.train = train
            self.val = val
            split_predict = [self.num_samples, len(self.train) - self.num_samples]
            self.predict, _ = random_split(self.train, split_predict) #get a small amount of train data for predict step
            #TODO: figure out how to obtain num_samples (breaking the dependency to the batch_size, but still getting shuffled data)
            n_samples = min(self.num_samples, len(self.predict))
            self.samples["train"] = next(iter(self.create_dataloader(self.predict, batch_size=n_samples))) #"reference" train samples always from predict dataloader
            n_samples = min(self.num_samples, len(self.val))
            self.samples["val"] = next(iter(self.create_dataloader(self.val, batch_size=n_samples)))
        if stage == 'test' or stage is None:
            if self.test is None:
                dataset_test = self.dataset_object(self.data_dir, train=False, transform=self.transform)
                self.test = dataset_test
            n_samples = min(self.num_samples, len(self.test))
            self.samples["test"] = next(iter(self.create_dataloader(self.test, batch_size=n_samples)))

        if stage is None:
            if dataset_train is None:
                dataset_train = self.dataset_object(self.data_dir, train=True, transform=self.transform)
            if dataset_test is None:
                dataset_test = self.dataset_object(self.data_dir, train=False, transform=self.transform)
            self.dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
            targets = torch.concat([torch.stack(dataset_train.targets), torch.stack(dataset_test.targets)]).tolist()
            self.target_contents = dataset_train.target_contents
            for key in self.target_contents.keys():
                if isinstance(self.target_contents[key], list):
                    self.target_contents[key].extend(dataset_test.target_contents[key])
                elif isinstance(self.target_contents[key], torch.Tensor):
                    self.target_contents[key] = \
                        torch.cat((self.target_contents[key], dataset_test.target_contents[key]))
                    self.target_contents[key] = self.target_contents[key].tolist()
                elif isinstance(self.target_contents[key], dict):
                    for k in self.target_contents[key].keys():
                        self.target_contents[key][k].extend(dataset_test.target_contents[key][k])
                else:
                    raise ValueError("Unsupported type for target_contents")
                if isinstance(self.target_contents[key], dict):
                    for k in self.target_contents[key].keys():
                        self.target_contents[key][k] = dict(zip(targets, self.target_contents[key][k]))
                else:
                    self.target_contents[key] = dict(zip(targets, self.target_contents[key]))

    def train_dataloader(self):
        loader = self.create_dataloader(self.train, batch_size=self.batch_size, shuffle=True)
        return loader

    # Double workers for val and test loaders since there is no backward pass and GPU computation is faster
    def val_dataloader(self):
        loader = self.create_dataloader(self.val, batch_size=self.batch_size, shuffle=False)
        return loader

    def predict_dataloader(self):
        loader = self.create_dataloader(self.predict, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        loader = self.create_dataloader(self.test, batch_size=self.batch_size, shuffle=False)
        return loader

    def dataset_dataloader(self):
        loader = self.create_dataloader(self.dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def create_dataloader(self, data, batch_size, shuffle=True):
        data_type = self.dataset_metadata.config['data_type']
        if data_type == 'graph':
            data_loader = GraphDataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)
        else:
            raise NotImplementedError("Data Module not currently implemented for non Graph Data.")
        return data_loader

    def find_label_ids(self, dataset, label_descriptor, value):
        ids = [i for i, x in enumerate(dataset.target_contents[label_descriptor]) if x == value]
        return ids

    @property
    def dataset_object(self):
        return NotImplementedError

    @property
    def images(self):
        try:
            images = self.target_contents['images'].to(torch.float)
        except KeyError as e:
            logger.warning("Images were not loaded in the data_loader")
            images = None

        return images

    @property
    def task_structures(self):
        return self.target_contents['task_structure']

class GridNavDataModule(CustomDataModule):

    def __init__(self, data_dir: str = "", batch_size: int = 32, num_samples: int = 2048,
                 transform=None, num_workers: int = 0, val_data: str = 'train', no_images=True,
                 held_out_tasks:List[str]=None,
                 **kwargs):
        super().__init__(data_dir, batch_size, num_samples, transform, num_workers, val_data, no_images, held_out_tasks,
                         **kwargs)

    @property
    def dataset_object(self):
        return GridNavDataset


class CaveEscapeDataModule(CustomDataModule):

    def __init__(self, data_dir: str = "", batch_size: int = 32, num_samples: int = 2048,
                 transform=None, num_workers: int = 0, val_data: str = 'train', no_images=True,
                 held_out_tasks:List[str]=None,
                 **kwargs):
        super().__init__(data_dir, batch_size, num_samples, transform, num_workers, val_data, no_images, held_out_tasks,
                         **kwargs)

    @property
    def dataset_object(self):
        return CaveEscapeDataset


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            data, targets = b
            yield (self.func(data), self.func(targets))

    @property
    def dataset(self):
        return self.dl.dataset