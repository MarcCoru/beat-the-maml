import os
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch

from torch.utils.data.sampler import RandomSampler
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from .utils import download_file, unzip, query_yes_no
import random
from tqdm import tqdm
from skimage import exposure

s1bands = ["S1VV", "S1VH"]
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11", "S2B12"]
bands = s1bands + s2bands

H5URL = "https://syncandshare.lrz.de/dl/fiDJwH3ZgzcoDts3srTT8XaA/sen12ms.h5"
CSVURL = "https://syncandshare.lrz.de/dl/fiHr4oDKXzPSPYnPRWNxAqnk/sen12ms.csv"
CSVSIZE = 47302099
H5SIZE = 115351475848



allregions = [143, 131, 77, 114, 41, 137, 127, 147, 44, 119, 133, 93, 72,
              53, 82, 66, 142, 86, 20, 7, 64, 26, 15, 45, 58, 128,
              57, 29, 129, 97, 4, 121, 132, 112, 25, 116, 55, 36, 105,
              140, 75, 104, 68, 124, 76, 71, 139, 107, 125, 87, 37, 6,
              84, 39, 79, 94, 35, 31, 40, 28, 80, 19, 59, 69, 148,
              109, 62, 106, 9, 83, 88, 113, 138, 43, 24, 61, 33, 63, 56, 52, 11, 85, 117, 89, 136, 115, 27, 42,
              78, 102, 47, 120, 100, 3, 30, 14, 1, 149, 144, 146,
              118, 126, 122, 145, 108, 8, 65, 101, 81, 134, 123, 103, 135,
              110, 95, 130, 49, 91, 17, 22, 32, 73, 90, 141, 21
              ]

trainregions = [57, 27, 77, 94, 61, 3, 142, 43, 79, 14, 39, 100, 56, 53, 147, 4, 15, 58, 112, 44, 124, 59, 114, 113, 71,
                125, 127, 146, 117, 33, 80, 11, 47, 9, 6, 29, 20, 35, 69, 24, 131, 19, 68, 104, 41, 66, 86, 75, 105,
                137, 120, 28, 143, 25, 129, 37, 93, 116, 45, 84, 133, 121, 62, 31, 52, 115, 132, 136, 82, 102, 7, 97,
                87, 149, 144]

valregions = [83, 64, 30, 138, 63, 128, 36, 85, 139, 140, 109, 40, 72, 78, 88, 42, 55, 26, 89, 119, 1, 106, 148, 107,
              76]

holdout_regions = [118, 126, 122, 145, 108, 8, 65, 101, 81, 134, 123, 103, 135,
                   110, 95, 130, 49, 91, 17, 22, 32, 73, 90, 141, 21]

# IGBP classes
IGBP_classes = [
    "Evergreen Needleleaf Forests",
    "Evergreen Broadleaf Forests",
    "Deciduous Needleleaf Forests",
    "Deciduous Broadleaf Forests",
    "Mixed Forests",
    "Closed (Dense) Shrublands",
    "Open (Sparse) Shrublands",
    "Woody Savannas",
    "Savannas",
    "Grasslands",
    "Permanent Wetlands",
    "Croplands",
    "Urban and Built-Up Lands",
    "Cropland Natural Vegetation Mosaics",
    "Permanent Snow and Ice",
    "Barren",
    "Water Bodies"
]

# simplified IGBP classes (DFC2020) Schmitt et al. 2020, Yokoya et al. 2020
IGBP_simplified_classes = [
    "Forests",
    "Shrubland",
    "Savanna",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban Build-up",
    "Snow Ice",
    "Barren",
    "Water"
]

IGBP_simplified_class_mapping = [
    0,  # Evergreen Needleleaf Forests
    0,  # Evergreen Broadleaf Forests
    0,  # Deciduous Needleleaf Forests
    0,  # Deciduous Broadleaf Forests
    0,  # Mixed Forests
    1,  # Closed (Dense) Shrublands
    1,  # Open (Sparse) Shrublands
    2,  # Woody Savannas
    2,  # Savannas
    3,  # Grasslands
    4,  # Permanent Wetlands
    5,  # Croplands
    6,  # Urban and Built-Up Lands
    5,  # Cropland Natural Vegetation Mosaics
    7,  # Permanent Snow and Ice
    8,  # Barren
    9,  # Water Bodies
]


def sen12ms(folder, shots, ways, shuffle=True, test_shots=None,
            seed=None, **kwargs):
    if test_shots is None:
        test_shots = shots

    dataset = Sen12MS(folder, num_classes_per_task=ways, min_samples_per_class=shots + test_shots,
                      min_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
                            num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset


def prepare_dataset(args, transform):
    dataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                      target_transform=None,
                      meta_split="train", shuffle=True)

    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     sampler=CombinationSubsetRandomSampler(dataset))

    valdataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                         target_transform=None,
                         meta_split="val", shuffle=True)

    valdataloader = BatchMetaDataLoader(valdataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=CombinationSubsetRandomSampler(valdataset))

    return dataloader, valdataloader


def prepare_regular_dataset(args, transform):
    dataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                      target_transform=None,
                      meta_split="train", shuffle=True)

    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     sampler=CombinationSubsetRandomSampler(dataset))

    valdataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                         target_transform=target_transform,
                         meta_split="val", shuffle=True)

    valdataloader = BatchMetaDataLoader(valdataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=CombinationSubsetRandomSampler(valdataset))

    return dataloader, valdataloader


class Sen12MS(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_split="train", transform=None,
                 target_transform=None, min_samples_per_class=None, min_classes_per_task=0, **kwargs):
        dataset = Sen12MSClassDataset(root, meta_split=meta_split, transform=transform,
                                      target_transform=target_transform, min_samples_per_class=min_samples_per_class,
                                      min_classes_per_task=min_classes_per_task, **kwargs)

        super(Sen12MS, self).__init__(dataset, num_classes_per_task, target_transform=target_transform)


class Sen12MSClassDataset(ClassDataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split="train",
                 transform=None, target_transform=None, class_augmentations=None, min_samples_per_class=None,
                 min_classes_per_task=None, simplified_igbp_labels=True):
        super(Sen12MSClassDataset, self).__init__(meta_train=meta_train,
                                                  meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                  class_augmentations=class_augmentations)
        print(f"Initializing {meta_split} meta-dataset")

        self.transform = transform
        self.target_transform = target_transform
        self.meta_test = meta_test
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.root = root
        self.h5file_path = os.path.join(self.root, "sen12ms.h5")
        self.paths_file = os.path.join(self.root, "sen12ms.csv")

        if not os.path.exists(self.paths_file) or \
            not os.path.exists(self.h5file_path) or \
            not os.path.getsize(self.paths_file) == CSVSIZE or \
            not os.path.getsize(self.h5file_path) == H5SIZE:

            if query_yes_no(f"No dataset found at {self.root}. Do you want to download (108GB)?"):
                print(f"downloading {CSVURL} to {self.paths_file}")
                download_file(CSVURL, self.paths_file, overwrite=True)
                print(f"downloading {H5URL} to {self.h5file_path}")
                download_file(H5URL, self.h5file_path, overwrite=True)
            else:
                import sys
                sys.exit()

        if self.meta_train or self.meta_split == "train":
            regions = trainregions
        elif self.meta_val or self.meta_split == "val":
            regions = valregions
        elif self.meta_test or self.meta_split == "test":
            regions = holdout_regions
        else:
            raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                 "meta_split must be in 'train','val','test'")

        self.regions = regions
        seasons = ["summer", "spring", "fall", "winter"]

        self.paths = pd.read_csv(self.paths_file, index_col=0)

        if simplified_igbp_labels:
            self.classes = IGBP_simplified_classes
        else:
            self.classes = IGBP_classes

        # list of all regions with classes
        counts = self.paths[["season", "region", "maxclass", "lcpath"]].groupby(
            by=["season", "region", "maxclass"]).count().reset_index()
        if min_samples_per_class is not None:
            mask = counts["lcpath"] > min_samples_per_class
            self._labels = counts.loc[mask][["season", "region", "maxclass"]]
            print(
                f"keeping {mask.sum()}/{len(counts)} region/class pairs with >{min_samples_per_class} samples per class")
        else:
            self._labels = counts[["season", "region", "maxclass"]]

        tasks_idxs = list()
        for region in regions:
            for season in seasons:
                mask = (self._labels["region"] == region) & (self._labels["season"] == season)
                task_idx = self._labels.reset_index(drop=True).index[mask].tolist()
                if len(task_idx) > min_classes_per_task:
                    tasks_idxs.append(task_idx)
        self.task_idxs = tasks_idxs
        print(
            f"keeping {len(tasks_idxs)}/{len(regions) * len(seasons)} regions/season pairs with >{min_classes_per_task} unique classes per region")

        """
        self.metadata = list()
        for idx in range(len(self.labels)):
            regionclass = self.labels.iloc[idx]
            selection_mask = (self.paths["region"] == regionclass.region) & \
                             (self.paths["maxclass"] == regionclass.maxclass)
            selected_paths = self.paths.loc[selection_mask]
            self.metadata.append(selected_paths[["lcpath", "s1path", "s2path", "npzpath","h5path","tile"]].values)
        """

        # speedup when querying from array in __get_item__
        self._labels = self._labels.values
        self._data = None

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def data(self):
        # if self._data is None:
        #    self._data = h5py.File(self.h5file_path, 'r')
        return None  # self._data

    @property
    def labels(self):
        return self._labels

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def __getitem__(self, idx):
        season, region, classname = self.labels[idx]
        subgroup = f"{season}/{region}/{classname.replace(' ', '_').replace('/', '_')}"
        return Sen12MSDataset(idx, self.h5file_path, subgroup, region, classname, self.transform, self.target_transform)


class Sen12MSDataset(Dataset):
    def __init__(self, index, h5file_path, group, region, classname, transform=None,
                 target_transform=None, debug=False):
        super(Sen12MSDataset, self).__init__(index)

        # TODO remove target_transform references
        if target_transform is not None:
            raise NotImplementedError("target_transform is not used anymore. define a transform function "
                                      "that takes (s1, s2, label) instead")

        # IGBP [2], and LCCS Land Cover, Land Use, and Surface Hydrology [3].

        self.h5file_path = h5file_path
        with h5py.File(h5file_path, 'r') as data:
            self.tiles = list(data[group].keys())
        self.group = group
        self.transform = transform
        self.target_transform = target_transform
        self.region = region
        self.classname = classname
        self.counter = 0
        self.debug = debug

    def __len__(self):
        return len(self.tiles)

    def load_tiff(self, lcpath, s1path, s2path):
        with rasterio.open(os.path.join(self.root, s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.root, s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.root, lcpath), "r") as src:
            lc = src.read(1)

        image = np.vstack([s1, s2]).swapaxes(0, 2)
        target = lc
        return image, target

    def _tiff2npz(self, lcpath, s1path, s2path, npzpath):
        """takes 50ms per loop"""
        image, target = self.load_tiff(lcpath, s1path, s2path)

        npzpath = os.path.join(self.root, npzpath)
        os.makedirs(os.path.dirname(npzpath), exist_ok=True)
        np.savez(npzpath, image=image, target=target)
        return image, target

    def load_npz(self, npzpath):
        with np.load(os.path.join(self.root, npzpath), allow_pickle=True) as f:
            image = f["image"]
            target = f["target"]
        return image, target

    def __getitem__(self, index):
        tile = self.tiles[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[self.group + "/" + tile + "/s2"][()]
            s1 = data[self.group + "/" + tile + "/s1"][()]
            label = data[self.group + "/" + tile + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        # if self.target_transform is not None:
        #    target = self.target_transform((target, cropxy))

        if self.debug:
            self.counter += 1
            print(f"{self.region:<4}, {self.classname:<50}, {self.counter:<20}")

        return image, target, self.group + "/" + tile


class AllSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, transform, classes=None, seasons=None):
        super(AllSen12MSDataset, self).__init__()

        self.transform = transform

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if fold == "train":
            regions = trainregions
        elif fold == "val":
            regions = valregions
        elif fold == "test":
            regions = holdout_regions
        elif fold == "all":
            regions = holdout_regions + valregions + trainregions
        else:
            raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                 "fold must be in 'train','val','test'")

        mask = self.paths.region.isin(regions)
        print(f"fold {fold} specified. Keeping {mask.sum()} of {len(mask)} tiles")
        self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        return image, target, path.h5path


class CombinationSubsetRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise ValueError()
        self.data_source = data_source

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        task_idxs = self.data_source.dataset.task_idxs
        num_tasks = len(task_idxs)

        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            idxs = task_idxs[random.choice(range(num_tasks))]
            if len(idxs) >= num_classes_per_task:
                yield tuple(random.sample(idxs, num_classes_per_task))
            else:
                raise ValueError(f"{num_classes_per_task} are not enough classes for task idxs {idxs}")

def get_classification_transform():
    def transform(s1, s2, label):
        s2 = s2 * 1e-4
        s1 = s1 * 1e-2

        input = np.vstack([s1, s2])

        igbp_label = np.bincount(label.reshape(-1)).argmax() - 1
        target = IGBP_simplified_class_mapping[igbp_label]

        if np.random.rand() < 0.5:
            input = input[:, ::-1, :]

        # horizontal flip
        if np.random.rand() < 0.5:
            input = input[:, :, ::-1]

        # rotate
        n_rotations = np.random.choice([0, 1, 2, 3])
        input = np.rot90(input, k=n_rotations, axes=(1, 2)).copy()

        if np.isnan(input).any():
            print("found nan in input! replacing with 0")
            input = np.nan_to_num(input)
        assert not np.isnan(target).any()

        return torch.from_numpy(input), target

    return transform

def sample_testtasks(data_root="/ssd/sen12ms128", ways=4, shots=2, num_batches = 200):
    transform = get_classification_transform()

    rgb_idx = [bands.index(b) for b in np.array(['S2B4','S2B3', 'S2B2'])]

    def rgb_transform(s1,s2,y):
        X,y = transform(s1,s2,y)

        rgb = np.swapaxes(X[rgb_idx,:,:].numpy(),0,2)
        rgb = exposure.rescale_intensity(rgb)
        rgb = exposure.adjust_gamma(rgb, gamma=0.8, gain=1)
        X = np.swapaxes(rgb,0,2)
        return torch.from_numpy(X),y

    ## entire dataset ! 100 GB. Only download (comment in) if needed
    #!wget https://syncandshare.lrz.de/dl/fiHr4oDKXzPSPYnPRWNxAqnk/sen12ms.csv -P {data_root}
    #!wget https://syncandshare.lrz.de/dl/fiDJwH3ZgzcoDts3srTT8XaA/sen12ms.h5 -P {data_root}

    dataset = sen12ms(data_root, shots=shots, ways=ways, transform=rgb_transform,
                      target_transform=None, meta_split="test", shuffle=True)

    dataloader = BatchMetaDataLoader(dataset, batch_size=1,
                                 shuffle=False, num_workers=0,
                                 sampler=CombinationSubsetRandomSampler(dataset))


    testtasks = list()
    stats = list()
    for batch_idx, batch in tqdm(enumerate(dataloader),total=num_batches, leave=True):
        testtasks.append((batch_idx,batch))
        stat = pd.DataFrame.from_records([ids[0].split("/") for ids in batch["train"][2]], columns=["season","region","class","tile"])
        stat["batch_idx"] = batch_idx
        stats.append(stat)
        if batch_idx >= num_batches-1:
            break
    stats = pd.concat(stats)
    return testtasks, stats


if __name__ == "__main__":

    root = "/"

    classes = [
        "Evergreen Needleleaf Forests",
        "Evergreen Broadleaf Forests",
        "Deciduous Needleleaf Forests",
        "Deciduous Broadleaf Forests",
        "Mixed Forests",
        "Closed (Dense) Shrublands",
        "Open (Sparse) Shrublands",
        "Woody Savannas",
        "Savannas",
        "Grasslands",
        "Permanent Wetlands",
        "Croplands",
        "Urban and Built-Up Lands",
        "Cropland Natural Vegetation Mosaics",
        "Permanent Snow and Ice",
        "Barren",
        "Water Bodies"
    ]

    width = 32
    height = 32  #
    from torchvision.transforms import Compose, ToTensor
    import random


    def randomCrop(img):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width]
        return img


    transform = Compose([
        randomCrop,
        ToTensor()]
    )


    # categorical = Categorical(num_classes=args.num_ways)
    def target_transform(target):
        if isinstance(target, str):
            return classes.index(target.replace("/", " "))
        elif isinstance(target, int):
            return target


    dataset = sen12ms("/data2/sen12ms/", shots=5, ways=5, transform=transform,
                      target_transform=target_transform,
                      meta_split="train")

    dataloader = BatchMetaDataLoader(dataset, batch_size=2,
                                     shuffle=False, num_workers=0)

    from tqdm import tqdm
    import torch

    # for i in tqdm(range(100), total=1000):
    #    dataset.sample_task()["train"][0]

    num_batches = 20

    with tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            train_inputs, train_targets = batch['train']

            test_inputs, test_targets = batch['test']

            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                pass
            if batch_idx >= num_batches:
                break
