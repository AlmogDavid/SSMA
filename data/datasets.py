import abc
import itertools
import os
from copy import deepcopy
from random import shuffle
from typing import Dict, Type, Optional, Callable

import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.graphproppred import Evaluator as GEvaluator
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import Evaluator as NEvaluator
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import LRGBDataset as LRGBDatasetPYG
from torch_geometric.datasets import TUDataset as TUDatasetPYG
from torch_geometric.datasets import ZINC as ZINCPYG
from torch_geometric.loader import ClusterData
from torch_geometric.loader import ClusterLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from utils import get_all_subclasses


class BaseDataset(abc.ABC):
    """
    Base class for all datasets
    """

    DATASETS_ROOT_DIR = os.path.join(os.path.dirname(__file__), "datasets_data")
    """ The root dir to hold the data of the datasets"""

    @abc.abstractmethod
    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes the metric of the dataset
        :param pred: The predicted values
        :param gt: The ground truth values
        """
        pass

    @abc.abstractmethod
    def metric_name(self) -> str:
        """
        Returns the name of the metric that the dataset uses
        """
        pass

    @abc.abstractmethod
    def metric_type(self) -> str:
        """
        Returns the type of metric that the dataset uses, whether it needs to be minimized or maximized
        Return one of the following:
        - 'minimize'
        - 'maximize'
        """
        pass

    @abc.abstractmethod
    def get_split(self, split_name: str):
        """
        Returns the split of the dataset
        :param split_name: The name of the split to return
        """
        pass

    @abc.abstractmethod
    def node_dim(self) -> int:
        """
        Returns the dimension of the node features
        """
        pass

    @abc.abstractmethod
    def edge_dim(self) -> int:
        """
        Returns the dimension of the edge features
        """
        pass

    @abc.abstractmethod
    def output_dim(self) -> int:
        """
        Returns the dimension of the output that the model should produce
        """
        pass

    def is_graph_level(self) -> bool:
        return True

    def set_current_fold(self, f: int) -> None:
        pass

    @staticmethod
    def get_all_ds() -> Dict[str, Type['BaseDataset']]:
        """
        Returns a mapping of the dataset names to the dataset classes
        """
        all_sons = get_all_subclasses(BaseDataset)
        # Remove abstract classes
        non_abstract_subclasses = [cls for cls in all_sons if not cls.__abstractmethods__ and cls not in abc.ABC.__subclasses__()]
        return {cls.__name__: cls for cls in non_abstract_subclasses}


class OGBGDataset(BaseDataset):
    """
    Class for the OGB datasets
    """

    def __init__(self, ds_name: str, metric_name: str, metric_type: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self._ds_name = ds_name
        self._metric_name = metric_name
        self._metric_type = metric_type
        self._ds = PygGraphPropPredDataset(name=self._ds_name, root=os.path.join(BaseDataset.DATASETS_ROOT_DIR,
                                                                                 self._ds_name + "_pe" if pre_transform is not None else ""),
                                           transform=transform, pre_transform=pre_transform)
        self._split_idx = self._ds.get_idx_split()
        self._evaluator = GEvaluator(name=self._ds_name)
        self._output_dim = self._ds.num_classes
        self._node_dim = self._ds.num_node_features
        self._edge_dim = self._ds.num_edge_features

    def get_split(self, split_name: str):
        return self._ds[self._split_idx[split_name]]

    def metric_name(self) -> str:
        return self._metric_name

    def metric_type(self) -> str:
        return self._metric_type

    def output_dim(self) -> int:
        return self._output_dim

    def node_dim(self) -> int:
        return self._node_dim

    def edge_dim(self) -> int:
        return self._edge_dim


class OGBG_MOLHIV(OGBGDataset):
    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__(ds_name='ogbg-molhiv', metric_name='rocauc', metric_type='maximize',
                         pre_transform=pre_transform)

    def output_dim(self) -> int:
        return 1

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(input=pred, target=gt.type(torch.float32))

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self._evaluator.eval({'y_true': gt, 'y_pred': pred})[self._metric_name])


class OGBG_MOLPCBA(OGBGDataset):
    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__(ds_name='ogbg-molpcba', metric_name='ap', metric_type='maximize', pre_transform=pre_transform)

    def output_dim(self) -> int:
        return 128

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten()
        gt = gt.flatten()
        valid = gt == gt
        pred = pred[valid]
        gt = gt[valid]
        return F.binary_cross_entropy_with_logits(input=pred, target=gt.float())

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self._evaluator.eval({'y_true': gt, 'y_pred': pred})[self._metric_name])


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


class OGBNDataset(BaseDataset, abc.ABC):
    """
    Class for the OGBN datasets
    """

    def __init__(self, ds_name: str, num_clusters: int, pre_transform: Optional[Callable] = None):
        self._ds_name = ds_name
        self._ds_dir = os.path.join(BaseDataset.DATASETS_ROOT_DIR,
                                    self._ds_name + "_pe" if pre_transform is not None else "")
        self._ds = PygNodePropPredDataset(name=self._ds_name, root=self._ds_dir)
        self._split_idx = self._ds.get_idx_split()
        self._evaluator = NEvaluator(name=self._ds_name)
        self._output_dim = self._ds.num_classes
        self._node_dim = self._ds.num_node_features
        self._metric_name = "accuracy"
        self._num_clusters = num_clusters
        self._splits = self._cluster_data(pre_transform)

    def _cluster_data(self, pre_transform: Optional[Callable] = None):
        save_dir = self._ds_dir + f"_clustered_{self._num_clusters}"
        os.makedirs(save_dir, exist_ok=True)

        full_data = self._ds[0]
        full_data.train_mask = torch.zeros(full_data.x.size(0), dtype=torch.bool, device=full_data.x.device)
        full_data.train_mask[self._split_idx["train"]] = True
        full_data.val_mask = torch.zeros(full_data.x.size(0), dtype=torch.bool, device=full_data.x.device)
        full_data.val_mask[self._split_idx["valid"]] = True
        full_data.test_mask = torch.zeros(full_data.x.size(0), dtype=torch.bool, device=full_data.x.device)
        full_data.test_mask[self._split_idx["test"]] = True

        full_data.edge_index = to_undirected(full_data.edge_index, None, num_nodes=full_data.x.size(0))[0]
        cluster_data = ClusterData(full_data,
                                   num_parts=self._num_clusters,
                                   save_dir=save_dir)
        loader = ClusterLoader(cluster_data)

        # Prepare splits
        train_split = []
        val_split = []
        test_split = []

        for data in tqdm(loader, desc="Preparing data"):
            data = pre_transform(data) if pre_transform is not None else data
            train_data = deepcopy(data)
            train_data.y[~train_data.train_mask] = -1
            val_data = deepcopy(data)
            val_data.y[~val_data.val_mask] = -1
            test_data = deepcopy(data)
            test_data.y[~test_data.test_mask] = -1

            train_split.append(train_data)
            val_split.append(val_data)
            test_split.append(test_data)

        return {"train": train_split, "valid": val_split, "test": test_split, 'all': [full_data]}

    def get_split(self, split_name: str):
        return self._splits[split_name]

    def metric_name(self) -> str:
        return self._metric_name

    def metric_type(self) -> str:
        return "maximize"

    def output_dim(self) -> int:
        return self._output_dim

    def node_dim(self) -> int:
        return self._node_dim

    def edge_dim(self) -> int:
        return 0

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        valid_gt = gt.flatten() >= 0
        return F.cross_entropy(input=pred[valid_gt], target=gt.flatten()[valid_gt])

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        valid_gt = gt.flatten() >= 0
        return torch.as_tensor(self._evaluator.eval({'y_true': gt[valid_gt], 'y_pred': pred[valid_gt].argmax(-1, keepdim=True)})["acc"])

    def is_graph_level(self) -> bool:
        return False


class OGBNArxiv(OGBNDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__(ds_name="ogbn-arxiv", num_clusters=128, pre_transform=pre_transform)


class OGBNProducts(OGBNDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__(ds_name="ogbn-products", num_clusters=1024, pre_transform=pre_transform)


class TUDataset(BaseDataset, abc.ABC):

    def __init__(self, ds_name: str, k: int = 10, pre_transform: Optional[Callable] = None):
        self._ds_name = ds_name
        ds = TUDatasetPYG(root=os.path.join(BaseDataset.DATASETS_ROOT_DIR,
                                            self._ds_name + "_pe" if pre_transform is not None else ""),
                          name=self._ds_name, pre_transform=pre_transform)
        self._node_dim = ds.num_features
        self._edge_dim = ds.num_edge_features
        self._output_dim = ds.num_classes

        # Create KFold
        self._f = 0
        ds = [d for d in ds]
        shuffle(ds)
        split_length = len(ds) // k
        splits = []
        for i in range(k):
            if i == k-1:
                splits.append(ds[split_length*i:])
            else:
                splits.append(ds[split_length*i:split_length*(i+1)])
        self._splits = splits

    def set_current_fold(self, f: int) -> None:
        assert 0 <= f < len(self._splits), f"Invalid fold: {f}"
        self._f = f

    def get_split(self, split_name: str):
        valid_idx = (self._f + 1) % len(self._splits)
        test_idx = self._f
        if split_name == "train":
            ds = itertools.chain.from_iterable([self._splits[i] for i in range(len(self._splits)) if i != valid_idx and i != test_idx])
            ds = list(ds)
        elif split_name == "valid":
            ds = self._splits[valid_idx]
        elif split_name == "test":
            ds = self._splits[test_idx]
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        return ds

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input=pred, target=gt)

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return MF.accuracy(preds=pred, target=gt, task="multiclass", num_classes=self.output_dim())

    def metric_name(self) -> str:
        return "accuracy"

    def metric_type(self) -> str:
        return "maximize"

    def node_dim(self) -> int:
        return self._node_dim

    def edge_dim(self) -> int:
        return self._edge_dim

    def output_dim(self) -> int:
        return self._output_dim


class MUTAG(TUDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("MUTAG", pre_transform=pre_transform)


class ENZYMES(TUDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("ENZYMES", pre_transform=pre_transform)


class PROTEINS(TUDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("PROTEINS", pre_transform=pre_transform)


class PTC_MR(TUDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("PTC_MR", pre_transform=pre_transform)


class IMDBB(TUDataset):

    def __init__(self, node_dim: int = 32, pre_transform: Optional[Callable] = None):
        super().__init__("IMDB-BINARY", pre_transform=pre_transform)
        # Set random features
        node_features = torch.randn(node_dim)
        self._node_dim = node_dim
        new_splits = []
        for s in self._splits:
            curr_new_split = []
            for d in s:
                d.x = node_features.unsqueeze(0).repeat(d.num_nodes, 1)
                curr_new_split.append(d)
            new_splits.append(curr_new_split)
        self._splits = new_splits


class ZINC(BaseDataset):

    def __init__(self, subset: bool = True, pre_transform: Optional[Callable] = None):
        ds_dir = os.path.join(BaseDataset.DATASETS_ROOT_DIR, "ZINC" + "_pe" if pre_transform is not None else "")
        self._train_ds = ZINCPYG(root=ds_dir, subset=subset, split="train", pre_transform=pre_transform)
        self._val_ds = ZINCPYG(root=ds_dir, subset=subset, split="val", pre_transform=pre_transform)
        self._test_ds = ZINCPYG(root=ds_dir, subset=subset, split="test", pre_transform=pre_transform)

        self._node_dim = 1
        self._edge_dim = 1

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        #return F.mse_loss(input=pred.flatten(), target=gt.flatten())
        return (pred.flatten() - gt.flatten()).abs().mean()

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return MF.mean_absolute_error(preds=pred.flatten(), target=gt.flatten())

    def metric_name(self) -> str:
        return "MAE"

    def metric_type(self) -> str:
        return "minimize"

    def get_split(self, split_name: str):
        if split_name == "train":
            return self._train_ds
        elif split_name == "valid":
            return self._val_ds
        elif split_name == "test":
            return self._test_ds
        else:
            raise ValueError(f"Invalid split: {split_name}")

    def node_dim(self) -> int:
        return self._node_dim

    def edge_dim(self) -> int:
        return self._edge_dim

    def output_dim(self) -> int:
        return 1


class LRGBDataset(BaseDataset):

    def __init__(self, name: str, pre_transform: Optional[Callable] = None):
        ds_dir = os.path.join(BaseDataset.DATASETS_ROOT_DIR, name + "_pe" if pre_transform is not None else "")
        self._train_ds = LRGBDatasetPYG(root=ds_dir, name=name, split="train", pre_transform=pre_transform)
        self._val_ds = LRGBDatasetPYG(root=ds_dir, name=name, split="val", pre_transform=pre_transform)
        self._test_ds = LRGBDatasetPYG(root=ds_dir, name=name, split="test", pre_transform=pre_transform)

        self._node_dim = self._train_ds.num_features
        self._edge_dim = self._train_ds.num_edge_features
        self._output_dim = self._train_ds.num_classes

    def get_split(self, split_name: str):
        if split_name == "train":
            return self._train_ds
        elif split_name == "valid":
            return self._val_ds
        elif split_name == "test":
            return self._test_ds
        else:
            raise ValueError(f"Invalid split name: {split_name}")

    def node_dim(self) -> int:
        return self._node_dim

    def edge_dim(self) -> int:
        return self._edge_dim

    def output_dim(self) -> int:
        return self._output_dim


class PeptidesFunc(LRGBDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("Peptides-func", pre_transform=pre_transform)

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(input=pred, target=gt)

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return MF.average_precision(preds=pred, target=gt.type(torch.int), task="multilabel", num_labels=self._train_ds.num_classes)

    def metric_name(self) -> str:
        return "ap"

    def metric_type(self) -> str:
        return "maximize"


class PeptidesStruct(LRGBDataset):

    def __init__(self, pre_transform: Optional[Callable] = None):
        super().__init__("Peptides-struct", pre_transform=pre_transform)

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input=pred, target=gt)

    def compute_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return MF.mean_absolute_error(preds=pred.flatten(), target=gt.flatten())

    def metric_name(self) -> str:
        return "MAE"

    def metric_type(self) -> str:
        return "minimize"
