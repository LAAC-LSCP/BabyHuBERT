#!/bin/bash
#SBATCH --job-name=preprocess_features          # Job name
#SBATCH --partition=prepost                    # Take a node from the 'gpu' partition
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --nodes=1            # Request N GPUs per machine

## for A100 
##SBATCH -C a100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

## for prepost partition
#SBATCH --cpus-per-task=1


#SBATCH --hint=nomultithread 
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --array=0-31%32
#SBATCH --output=logs/slurm-%j-%a-gen_labels.out 
##SBATCH --output=logs/slurm-%j-%a-train_kmeans.out 
#module load arch/a100


export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load sox/14.4.2

source .venv/bin/activate


srun uv run preprocess.py -gl --num-shards-kmeans 6 --feat-type baby-hubert-175s --layer-index 7 --num-rank 32 --num-cluster 500


