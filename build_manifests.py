from glob import glob
import torch
import os
import torchaudio
from tqdm import tqdm
import tarfile
import io
import csv
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.feature_utils import dump_features, extract_feature_hubert, get_shard_range
from utils.common_utils import build_manifest_tar, read_from_archive
from pathlib import Path
import time


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="The directory to store the experiment outputs.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
    for split in ["train"]:
        manifest = build_manifest_tar(args.path + "archive_"+split+"_"+str(rank)+".tar", file_extension=".pt")
        manifest.to_csv(args.path + "manifest_"+split+"_"+str(rank)+".csv", index=False)
        