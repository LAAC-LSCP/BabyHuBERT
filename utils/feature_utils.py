#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import io
import tarfile
import torch
import torchaudio
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from .common_utils import _get_feat_lens_paths, read_from_archive, build_manifest_tar
import pandas as pd
import time 
_LG = logging.getLogger(__name__)
_DEFAULT_DEVICE = torch.device("cpu")

map_models = {
    "hubert_base":torchaudio.models.hubert_pretrain_base,
    "hubert_large":torchaudio.models.hubert_pretrain_large,
    #"wavlm_base":torchaudio.models.hubert_pretrain_base,
    "wavlm_base_plus":torchaudio.models.hubert_pretrain_base,
    #"wavlm_large":torchaudio.models.hubert_pretrain_base
}

map_features = {
    "hubert_base":torchaudio.pipelines.HUBERT_BASE,
    "hubert_large":torchaudio.pipelines.HUBERT_LARGE,
    "wavlm_base":torchaudio.pipelines.WAVLM_BASE,
    "wavlm_base_plus":torchaudio.pipelines.WAVLM_BASE_PLUS,
    "wavlm_large":torchaudio.pipelines.WAVLM_LARGE
}

def get_shard_range(num_lines: int, num_rank: int, rank: int) -> Tuple[int, int]:
    r"""Get the range of indices for the current rank in multi-processing.
    Args:
        num_lines (int): The number of lines to process.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        rank (int): The rank in the multi-processing.

    Returns:
        (int, int):
        int: The start index for the current rank.
        int: The end index for the current rank.
    """
    assert 1 <= rank <= num_rank, f"invalid rank/num_rank {rank}/{num_rank}"
    assert num_lines > 0, f"Found {num_lines} files, make sure you specify the correct root directory"
    start = round(num_lines / num_rank * (rank - 1))
    end = round(num_lines / num_rank * rank)
    _LG.info(f"rank {rank} of {num_rank}, process {end-start} " f"({start}-{end}) out of {num_lines}")
    return start, end


def extract_feature_mfcc(
    path: str,
    device: torch.device,
    sample_rate: int,
) -> Tensor:
    r"""Extract MFCC features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    feature_extractor = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "center": False}
    ).to(device)
    waveform = waveform[0].to(device)
    mfccs = feature_extractor(waveform)  # (freq, time)
    deltas = torchaudio.functional.compute_deltas(mfccs)
    ddeltas = torchaudio.functional.compute_deltas(deltas)
    concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
    feat = concat.transpose(0, 1)  # (time, freq)
    return feat


def extract_feature_hubert(
    manifest_row,
    wavs_archive,
    feat_archive,
    device: torch.device,
    sample_rate: int,
    model: Module,
    layer_index: int,
    rank:int
) -> Tensor:
    r"""Extract HuBERT features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.
        model (Module): The loaded ``HuBERTPretrainModel`` model.
        layer_index (int): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output).

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform = torch.load(read_from_archive(manifest_row["path"], manifest_row["byte_offset"], manifest_row["byte_size"] , wavs_archive))
    if waveform.shape[1] != manifest_row["num_frames"]:
        with open("./exp/problems/prob_"+str(rank)+".csv", "a") as problem:
            problem.write(manifest_row["path"]+ "," +   str(waveform.shape[1]) + "," +  str(manifest_row["num_frames"])  + "," + str(manifest_row["byte_offset"])+ "," + str(manifest_row["byte_size"])+ "\n")
            return
    with torch.inference_mode():

        feat = model.wav2vec2.extract_features(waveform.to(device), num_layers=layer_index)[0][-1][0]  # (time, feat_dim)

        buffer = io.BytesIO()
        torch.save(feat.cpu(), buffer)
        buffer.seek(0)
        tar_info = tarfile.TarInfo(name=manifest_row["path"])
        buffed = buffer.getbuffer().nbytes
        tar_info.size = buffed
        buffer.seek(0)
        feat_archive.addfile(tar_info,buffer)

def _load_state(model: Module, checkpoint_path: Path, device=_DEFAULT_DEVICE) -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model


def dump_features(
    archive_dir: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "mfcc",
    layer_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    sample_rate: int = 16_000,
    csv_mapping: Union[str, Path] = None,
) -> None:
    r"""Dump the feature tensors given a ``.tsv`` file list. The feature and lengths tensors
        will be stored under ``out_dir`` directory.
    Args:
        tsv_file (str or Path): The path of the tsv file.
        out_dir (str or Path): The directory to store the feature tensors.
        split (str): The split of data. Options: [``train``, ``valid``].
        rank (int): The rank in the multi-processing.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        feature_type (str, optional): The type of the desired feature. Options: [``mfcc``, ``hubert``].
            (Default: ``mfcc``)
        layer_index (int or None, optional): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output). Only active when ``feature_type``
            is set to ``hubert``. (Default: ``None``)
        checkpoint_path(Path or None, optional): The checkpoint path of ``torchaudio.models.HuBERTPretrainModel``.
            Only active when ``feature_type`` is set to ``hubert``. (Default: ``None``)
        sample_rate (int, optional): The sample rate of the audio. (Default: ``16000``)

    Returns:
        None
    """
    if feature_type != "mfcc" and layer_index is None:
        assert ValueError("Please set the layer_index for HuBERT feature.")
    
    out_dir = Path(out_dir)

    if feature_type != "mfcc":
        print("loading ", feature_type)
        if feature_type == "baby-hubert-175s":
            model = torchaudio.models.hubert_pretrain_base(num_classes=500)
            model = _load_state(model, "./exp_iter2_B175/checkpoints_longforms_hubert_pretrain_base/epoch=45-step=400000.ckpt")
            model.wav2vec2.to(device)
        else:

            model = map_models[feature_type]()
            #model.to(device)
            bundle = map_features[feature_type]
            hubert = bundle.get_model()
            model.wav2vec2 = hubert.to(device)
        #_LG.info(f"Finished loading hubert")

    manifest = pd.read_csv(archive_dir + "manifest_"+split+"_"+ str(rank)+".csv")
    with tarfile.open(out_dir / ("archive_"+split+"_"+str(rank)+".tar"), "a") as tar_archive:
        with open(archive_dir + "archive_"+split+"_"+ str(rank)+".tar", "rb") as archive_path:
            for i,row in tqdm(manifest.iterrows(), total=len(manifest.index)):

                if row["num_frames"] < 500:
                    _LG.info(f"{row["path"]}, {row["num_frames"]}")
                    continue
                if feature_type == "mfcc":
                    feature = extract_feature_mfcc(path, device, sample_rate)
                else:
                    #to modify for inference with checkpoint model 
                    extract_feature_hubert(row, archive_path, tar_archive, device, 16000, model, layer_index, rank)
    _LG.info(f"Finished dumping features for rank {rank} of {num_rank} successfully")