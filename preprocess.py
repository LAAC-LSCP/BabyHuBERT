#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation
"""
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import torch
from utils import create_tsv, dump_features, get_km_label, learn_kmeans
from utils.common_utils import build_manifest_tar
from tqdm import tqdm
import os

def _init_logger(debug=False):
    message_fmt = "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument("--dataset", default="librispeech", type=str, choices=["librispeech", "librilight", "longforms"])
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./archive_dataset/",
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num-rank", default=5, type=int)
    parser.add_argument("--feat-type", default="mfcc", type=str)
    parser.add_argument("--feat-iter", default="1", type=str)

    parser.add_argument("--csv-mapping", default=None, type=Path, help="The file mapping sample:onset,offset,audio_file,annotation_file")
    parser.add_argument(
        "--layer-index",
        default=6,
        type=int,
        help="The layer index in HuBERT model for feature extraction. (``1`` means the first layer output)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument("--use-gpu", default=False, type=bool)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("./exp_iter2_B175"),
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num-cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="The percent of data for KMeans clustering. If negative, use all data. (Default: -1)",
    )
    parser.add_argument("-gf",action="store_true",help="generate_features")
    parser.add_argument("-lk",action="store_true",help="learn_kmeans")
    parser.add_argument("-gl",action="store_true",help="generate_labels")
    parser.add_argument(
        "--num-shards-kmeans",
        default=6,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )

    args = parser.parse_args()
    return args


def main(args):
    _init_logger(args.debug)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
        #args.gpu = args.rank % torch.cuda.device_count()

    print("RANK  :", args.rank)

    if not args.exp_dir.exists():
        args.exp_dir.mkdir()
    if args.feat_type == "mfcc":
        data_dir = args.exp_dir / "data" / "mfcc"
    else:
        data_dir = args.exp_dir / "data" / f"{args.feat_type}_{args.feat_iter}_{args.layer_index}"
    data_dir.mkdir(parents=True, exist_ok=True)

    feat_dir = data_dir / "feat"
    km_dir = data_dir / "km_model"
    label_dir = data_dir / "label"
    archive_dir = args.root_dir

    if not feat_dir.exists():
        feat_dir.mkdir()
    
    # Generate features
    if args.gf:
        for split in ["train"]:
            dump_features(
                    archive_dir,
                    feat_dir,
                    split,
                    args.rank,
                    args.num_rank,
                    torch.device("cuda"),
                    args.feat_type,
                    args.layer_index,
                    args.checkpoint_path,
                    16_000,
                    args.csv_mapping
                )
            
            
    #Create Archive manifests 

    # Fit KMeans clustering model
    #Kmeans needs to be ran in only one job with no ranks
    if args.lk:
        learn_kmeans(
            feat_dir,
            "train",
            args.num_shards_kmeans,# number of shards to put learn from
            km_dir,
            args.num_cluster,
            args.percent,
        )
    

    # Predict labels for MFCC or HuBERT features
    if args.gl:
        for split in ["train"]:
            get_km_label(
                feat_dir,
                km_dir,
                label_dir,
                split,
                args.rank+1,
                args.num_rank,
                torch.device("cpu"),
                args.num_shards_kmeans
            )


if __name__ == "__main__":
    main(_parse_args())
