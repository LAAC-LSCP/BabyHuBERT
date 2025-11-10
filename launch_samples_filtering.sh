#!/bin/bash
#SBATCH --job-name=preprocess_features          # Job name
#SBATCH --partition=prepost                    # Take a node from the 'gpu' partition
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --nodes=1            # Request N GPUs per machine


#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1
##SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j-%a-samples.out 

##module load arch/a100


export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load sox/14.4.2

source .venv/bin/activate

srun uv run preprocess_samples.py ./datasets/all_samples_filtered.csv ./samples/samples_padded_2on_2off.csv --min-duration-on=2.0 --min-duration-off=2.0

