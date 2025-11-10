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


if __name__ == "__main__":
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        rank = int(os.environ['SLURM_ARRAY_TASK_ID'])

    scratch = "/archive_dataset/"
    splits = glob("./datasets/train_set.csv")
    path_datasets = "./datasets/"
    print(splits)
    for split in splits :
        str_split = split.split("/")[-1].split("_")[0]
        print("launching : ", str_split)
        with open(split, "r") as f:
            root = f.readline().strip()
            lines = [line for line in f]
            start, end = get_shard_range(len(lines), 32, rank+1)
            rank_lines = lines[start:end]
            print("to archive : ", start, " to ", end, " sum : ", end-start )
            with tarfile.open(scratch + f"archive_"+str_split+"_"+str(rank)+".tar", "a") as tar_archive:
                start_time= time.time()
                last_time = start_time
                for i,line in enumerate(rank_lines):
                    if i%1000 == 0:
                        curr_time = time.time()
                        print(i, curr_time - start_time, curr_time - last_time)
                        last_time = curr_time 
                    path, onset, offset  = line.replace("\n", "").split(",")
                    offset = str(int(onset) + int(offset))
                    waveform, sr = torchaudio.load(path_datasets + path, frame_offset=int(onset), num_frames=int(offset)-int(onset), backend="soundfile")
                    buffer = io.BytesIO()
                    torch.save(waveform, buffer)
                    splits = path.split(".")
                    buffer.seek(0)
                    tar_info = tarfile.TarInfo(name=splits[0] + "_" + str(onset) + "_" + str(offset) +".pt")
                    buffed = buffer.getbuffer().nbytes
                    tar_info.size = buffed
                    buffer.seek(0)
                    tar_archive.addfile(tar_info,buffer)




