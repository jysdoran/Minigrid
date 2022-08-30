import logging

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple, Dict

import torch
import pytorch_lightning as pl
import dgl

from torch.utils.data import random_split
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from dgl.dataloading import GraphDataLoader

logger = logging.getLogger(__name__)

class GridNav_Dataset(VisionDataset):
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
    train_list = [
        ["batch_0.data", "TODO"],
        ["batch_1.data", "TODO"],
        ["batch_2.data", "TODO"],
        ["batch_3.data", "TODO"],
        ["batch_4.data", "TODO"],
        ["batch_5.data", "TODO"],
        ["batch_6.data", "TODO"],
        ["batch_7.data", "TODO"],
        ["batch_8.data", "TODO"],
        ["batch_9.data", "TODO"],
    ]

    test_list = [
        ["test_batch.data", "TODO"],
        ["test_batch_90.data", "TODO"],
        ["test_batch_91.data", "TODO"],
    ]
    meta = {
        "filename": "dataset.meta",
        "md5": "ceb1fb9aaface2c9669a3914ecf7d30a",
        #"label_descriptors": "label", # TODO: figure out the structure of "label"
        "data_dim": "data_dim",
        "task:": "task_type",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._load_meta()

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            pickled_data = self.train_list
        else:
            pickled_data = self.test_list

        self.data: Any = []
        self.targets = []
        self.target_contents: Dict = None
        self.batches_metadata: Any = []

        # now load the picked numpy arrays
        for file_name, checksum in pickled_data:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            try:
                if self.data_type == 'graph':
                    if not os.path.exists(file_path): raise FileNotFoundError()
                    graphs, labels = dgl.load_graphs(file_path)
                    self.data.extend(graphs)
                    self.targets.extend(labels['labels'])
                    file_path += '.meta'
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    # note: will end loop prematurely if exeption is thrown, not sure this is the best pattern
                    # as behavior is order dependent
                    try:
                        self.batches_metadata.append(entry["batch_meta"])
                        if self.target_contents is None:
                            self.target_contents = entry["label_contents"]
                        else:
                            try:
                                for key in entry["label_contents"].keys():
                                    if isinstance(entry["label_contents"][key], list):
                                        self.target_contents[key].extend(entry["label_contents"][key])
                                    elif isinstance(entry["label_contents"][key], torch.Tensor):
                                        self.target_contents[key] = \
                                            torch.cat((self.target_contents[key], entry["label_contents"][key]))
                                    else:
                                        raise ValueError("Unsupported type for target_contents")
                            except KeyError as e:
                                raise KeyError(f"{e} not found in {self.target_contents.keys()}. Mismatch in label contents "
                                               f"across batch files")

                        # these ones will usually be skipped
                        self.data.extend(entry["data"])
                        self.targets.extend(entry["labels"])
                    except KeyError as e:
                        pass
            except FileNotFoundError as e:
                pass

        if not self.data:
            raise FileNotFoundError("Dataset not found at specified location.")

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        # if not check_integrity(path, self.meta["md5"]): #TODO: reenable
        #     raise RuntimeError("Dataset metadata file not found or corrupted.")
        with open(path, "rb") as infile:
            self.dataset_metadata = pickle.load(infile, encoding="latin1")
            self.label_descriptors = self.dataset_metadata["label_descriptors"]
            self.data_type = self.dataset_metadata["data_type"]
        self.label_to_idx = {_class: i for i, _class in enumerate(self.label_descriptors)} #TODO: decide if we replace by self.label_to_idx

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


class GridNavDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 32, num_samples: int = 2048,
                 transform=None, num_workers: int = 0, **kwargs):
        super().__init__()
        #sampler = dgl.dataloading.GraphDataLoader()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.transform = transform
        self.samples = {}
        self.dataset = None
        self.num_workers = num_workers
        logger.info("Initializing Gridworld Navigation DataModule")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset_full = GridNav_Dataset(self.data_dir, train=True, transform=self.transform)
            split_size = [int(0.9 * len(dataset_full)), len(dataset_full) - int(0.9 * len(dataset_full))]
            train, val = random_split(dataset_full, split_size)
            self.dataset = train.dataset
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
            self.test = GridNav_Dataset(self.data_dir, train=False, transform=self.transform)
            n_samples = min(self.num_samples, len(self.test))
            self.samples["test"] = next(iter(self.create_dataloader(self.test, batch_size=n_samples)))

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

    def create_dataloader(self, data, batch_size, shuffle=True):
        data_type = self.dataset.dataset_metadata['data_type']
        if data_type == 'graph':
            data_loader = GraphDataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)
        else:
            raise NotImplementedError("Data Module not currently implemented for non Graph Data.")
        return data_loader


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