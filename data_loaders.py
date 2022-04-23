import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset

class Maze_Dataset(VisionDataset):
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

    base_folder = "test_dataset"
    train_list = [
        ["batch_0.data", "TODO"],
    ]

    test_list = [
        ["test_batch", "TODO"],
    ]
    meta = {
        "filename": "dataset.meta",
        "md5": "ceb1fb9aaface2c9669a3914ecf7d30a",
        #"label_descriptors": "label", # TODO: figure out the structure of "label"
        "data_dim": "maze_size",
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

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            pickled_data = self.train_list
        else:
            pickled_data = self.test_list

        self.data: Any = []
        self.targets = []
        self.target_contents: Any = []
        self.batches_metadata: Any = []

        # now load the picked numpy arrays
        for file_name, checksum in pickled_data:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.batches_metadata.append(entry["batch_meta"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                if "label_contents" in entry:
                    self.target_contents.append(entry["label_contents"])

        self.data = np.vstack(self.data)

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted.")
        with open(path, "rb") as infile:
            self.dataset_metadata = pickle.load(infile, encoding="latin1")
            self.label_descriptors = self.dataset_metadata["label_descriptors"]
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
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

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