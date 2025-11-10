#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import Tuple

import torch
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
import numpy as np
from glob import glob
from .common_utils import _get_feat_lens_paths, _get_model_path, read_from_archive
import tarfile
import mmap
import io
import pandas as pd
from.feature_utils import get_shard_range
from tqdm import tqdm
_LG = logging.getLogger(__name__)


def load_feature(
    feat_dir: Path,
    split: str,
    num_rank: int,
    percent: float,
) -> Tuple[Tensor, Tensor]:
    r"""Loading features from pre-saved `.pt` files.
    Args:
        feat_dir (Path): The directory that stores the feature files.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        percent (float): The percent of data for training k-means model. If negative, use all data for training.

    Returns:
        (Tensor, Tensor)
        Tensor: The concatenated feature tensor of shape `(frame, feature_dim)`.
        Tensor: The lengths tensor of shape `(num_utterance,)`.
    """
    path = feat_dir
    feats = []
    lens = []

    # How many frames to pre allocate ?
    tot_frames = 0
    tot_features = 0
    for rank in range(1,num_rank + 1):
        manifest = pd.read_csv(path / ("manifest_"+split+"_"+str(rank-1)+".csv"))   
        tot_frames += sum(manifest["num_frames"])
        tot_features += sum(manifest["tensor_len"])


    print("total number of frames : ", tot_frames)
    print("total number of features : ", tot_features)
    
    # Pre allocate array of features
    feats = np.zeros((tot_features,768), dtype=np.float32)
    offset = 0
    for rank in range(1, num_rank + 1):

        manifest = pd.read_csv(path / ("manifest_"+split+"_"+str(rank-1)+".csv"))
        with open(path / ("archive_"+split+"_"+ str(rank-1)+".tar"), "rb") as archive_path:
            for i,row in tqdm(manifest.iterrows(), total=len(manifest.index)):
                feat = torch.load(read_from_archive(row["path"], row["byte_offset"], row["byte_size"] , archive_path))
                if percent < 1:
                    # Write directly in the contiguous pre allocated array
                    feats[offset:offset+feat.shape[0]] = feat.cpu().numpy()
                else:
                    # Not tested
                    offsets = [0] + torch.cumsum(length, dim=0, dtype=torch.int).tolist()
                    nsample = int(length.shape[0] * percent)
                    indices = torch.randperm(length.shape[0])[0:nsample]
                    indices = torch.sort(indices)[0]
                    mask = []
                    for i in range(indices.shape[0]):
                        index = indices[i]
                        mask += list(range(offsets[index], offsets[index] + length[index]))
                    mask = torch.tensor(mask, dtype=torch.int)
                    feat = torch.index_select(feat, 0, mask)
                    feats[offset:offset+feat.shape[0]] = feat.cpu().numpy()
                offset += feat.shape[0]
    return feats, lens


def learn_kmeans(
    feat_dir: Path,
    split: str,
    num_rank: int,
    km_dir: Path,
    n_clusters: int,
    percent: float = -1,
    init: str = "k-means++",
    max_iter: int = 100,
    batch_size: int = 10000,
    tol: float = 0.0,
    n_init: int = 20,
    reassignment_ratio: float = 0.0,
    max_no_improvement: int = 100,
) -> None:
    r"""Build and train the KMeans clustering model. The model is saved in "{km_dir}/model.pt"
    Args:
        feat_dir (Path): The directory that stores the feature files.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        km_dir (Path): The directory to store the KMeans clustering model.
        n_clusters (int): The number of clusters.
        percent (float): The percent of data for training k-means model.
            If negative, use all data for training. (Default: -1)
        init (str, optional): Method for initialization. Options: [``k-means++``, ``random``].
            (Default: ``k-means++``)
        max_iter (int, optional): Maximum number of iterations over the complete dataset. (Default: 100)
        batch_size (int, optional): Batch size for training the KMeans clustering model. (Default: 10000)
        tol (float, optional): Control early stopping based on the relative center changes as measured by a smoothed,
            variance-normalized of the mean center squared position changes. (Default: 0.0)
        n_init (int, optional): Number of random initializations that are tried. (Default: 20)
        reassignment_ratio (float, optional): Control the fraction of the maximum number of counts for a center
            to be reassigned. A higher value means that low count centers are more easily reassigned. (Default: 0.0)
        max_no_improvement (int, optional): Control early stopping based on the consecutive number of mini batches
            that does not yield an improvement on the smoothed inertia. (Default: 100)

    Returns:
        None
    """
    if not km_dir.exists():
        km_dir.mkdir()

    km_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=0,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )
    _LG.info("Starting Loading the KMeans clustering model")
    feats, _ = load_feature(
        feat_dir,
        split,
        num_rank,
        percent,
    )
    _LG.info("Starting training the KMeans clustering model")
    km_model.fit(feats)
    km_path = km_dir / ("model_"+str(num_rank)+"s.pt")
    import joblib

    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feats) / len(feats)
    _LG.info("Total intertia: %.5f", inertia)
    _LG.info("Finished training the KMeans clustering model successfully")


class ApplyKmeans:
    def __init__(self, km_path, device):
        import joblib

        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=1).cpu()


def get_km_label(
    feat_dir: Path,
    km_dir: Path,
    label_dir: Path,
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    shards: int
) -> None:
    r"""Predict the labels by the KMeans clustering model.
    Args:
        feat_dir (Path): The directory that stores the dumped features.
        km_dir (Path): The directory that stores the KMeans model.
        label_dir (Path): The directory to save the predicted labels.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
    Returns:
        None
    """
    if not label_dir.exists():
        label_dir.mkdir()

    km_path = _get_model_path(km_dir, shards)

    apply_kmeans = ApplyKmeans(km_path, device)
    
    path = feat_dir
    manifest = pd.read_csv(path / ("manifest_"+split+"_"+str(rank-1)+".csv"))
    # with append new archive 
    with tarfile.open(label_dir / ("archive_"+split+"_"+str(rank-1)+".tar"), "a") as labels_archive:
        with open(path / ("archive_"+split+"_"+ str(rank-1)+".tar"), "rb") as archive_path:
            for i,row in tqdm(manifest.iterrows(), total=len(manifest.index)):
                feat = torch.load(read_from_archive(row["path"], row["byte_offset"], row["byte_size"] , archive_path))
                labels = apply_kmeans(feat.to(device))
                buffer = io.BytesIO()
                torch.save(labels, buffer)
                buffer.seek(0)
                tar_info = tarfile.TarInfo(name=row["path"])
                buffed = buffer.getbuffer().nbytes
                tar_info.size = buffed
                buffer.seek(0)
                labels_archive.addfile(tar_info,buffer)
            
    _LG.info("Finished predicting labels successfully")
