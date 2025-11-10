#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
"""
Data pre-processing: create tsv files for training (and valiation).
"""
import logging
import re
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import tarfile
import mmap
from io import BytesIO

_LG = logging.getLogger(__name__)

def read_from_archive(archive: Path | str, offset: int, file_size: int, opened_archive = None) -> BytesIO:
    if opened_archive:
        with mmap.mmap(opened_archive.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            return BytesIO(mmap_o[offset : offset + file_size])
    with Path(archive).open("rb") as path, mmap.mmap(path.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
        return BytesIO(mmap_o[offset : offset + file_size])

class CompressedArchiveError(Exception):
    """Archive must be uncompressed to be read."""

def _manifest_from_bytes_info(archive: Path | str, bytes_info: dict[str, tuple[int | None, int, int]]) -> pd.DataFrame:
    manifest = pd.DataFrame.from_dict(bytes_info, orient="index", columns=["num_frames", "tensor_len", "byte_offset", "byte_size"])
    manifest["archive"] = archive
    manifest.index.name = "path"
    manifest["fileid"] = manifest.index.map(lambda x: Path(x).stem)
    return manifest.reset_index()[["fileid", "path", "num_frames", "tensor_len", "archive", "byte_offset", "byte_size"]]

def build_manifest_tar(path: Path | str, file_extension: str = ".wav", *, read_frames: bool = True) -> pd.DataFrame:
    if not tarfile.is_tarfile(path):
        raise ValueError(path)
    print("opening archive")
    with tarfile.open(path, mode="r") as tar_file:
        if tar_file.mode != "r":
            raise CompressedArchiveError
        infolist = tar_file.getmembers()
    print("starting iteration members archive")
    bytes_info = {}
    with Path(path).open("rb") as archive_desc:
        for info in tqdm(infolist):
            if info.isdir() or not info.name.endswith(file_extension):
                continue
            data = read_from_archive(path, info.offset_data, info.size, archive_desc)
            if file_extension == ".pt":
                splits = info.name.split(".")[0].split("_")
                num_frames = int(splits[-1]) - int(splits[-2]) if read_frames else None
                tensor_len = torch.load(data).shape[0] if read_frames else None
            else:
                num_frames = torchaudio.info(data).num_frames if read_frames else None
            bytes_info[info.name] = (num_frames, tensor_len, info.offset_data, info.size)
    if not bytes_info:
        raise NoAudioFileError(path, file_extension)
    return _manifest_from_bytes_info(path, bytes_info)


def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    dataset: str = "librispeech",
    valid_percent: float = 0.01,
    seed: int = 0,
    extension: str = "flac",
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        dataset (str, optional): The dataset to use. Options:
            [``librispeech``, ``libri-light``]. (Default: ``librispeech``)
        valid_percent (float, optional): The percentage of data for validation. (Default: 0.01)
        seed (int): The seed for randomly selecting the validation files.
        extension (str, optional): The extension of audio files. (Default: ``flac``)

    Returns:
        None
    """
    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(seed)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = open(out_dir / f"{dataset}_valid.tsv", "w") if valid_percent > 0 else None
    search_pattern = ".*train.*"
    with open(out_dir / f"{dataset}_train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        for fname in root_dir.glob(f"**/*.{extension}"):
            if re.match(search_pattern, str(fname)):
                frames = torchaudio.info(fname).num_frames
                dest = train_f if torch.rand(1) > valid_percent else valid_f
                print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
    if valid_f is not None:
        valid_f.close()
    _LG.info("Finished creating the file lists successfully")


def _get_feat_lens_paths(feat_dir: Path, split: str, rank: int, num_rank: int) -> Tuple[Path, Path]:
    r"""Get the feature and lengths paths based on feature directory,
        data split, rank, and number of ranks.
    Args:
        feat_dir (Path): The directory that stores the feature and lengths tensors.
        split (str): The split of data. Options: [``train``, ``valid``].
        rank (int): The rank in the multi-processing.
        num_rank (int): The number of ranks for multi-processing in feature extraction.

    Returns:
        (Path, Path)
        Path: The file path of the feature tensor for the current rank.
        Path: The file path of the lengths tensor for the current rank.
    """
    feat_path = feat_dir / f"{split}_{rank}_{num_rank}.pt"
    len_path = feat_dir / f"len_{split}_{rank}_{num_rank}.pt"
    return feat_path, len_path


def _get_model_path(km_dir: Path, shards: int) -> Path:
    r"""Get the file path of the KMeans clustering model
    Args:
        km_dir (Path): The directory to store the KMeans clustering model.

    Returns:
        Path: The file path of the model.
    """
    return km_dir / ("model_"+ str(shards) +"s.pt")


def _get_id2label() -> Dict:
    """Get the dictionary that maps indices of ASR model's last layer dimension to the corresponding labels."""
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    labels = bundle.get_labels()
    return {i: char.lower() for i, char in enumerate(labels)}


def _get_label2id() -> Dict:
    """Get the dictionary that maps the labels to the corresponding indices in ASR model's last dimension."""
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    labels = bundle.get_labels()
    return {char: i for i, char in enumerate(labels)}
