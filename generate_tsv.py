import torchaudio
from pathlib import Path
from tqdm import tqdm

root_dir = Path("/scratch2/tkunze/data/baby_train/wav/")

print("starting")
dest = open("./exp/data/custom-hubert_6/tsv/data_train.tsv", "w")
for fname in tqdm(root_dir.glob(f"*.wav")): 
    print(fname)
    frames = torchaudio.info(fname).num_frames
    print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)