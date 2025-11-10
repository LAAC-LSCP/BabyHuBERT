#!/bin/bash
#SBATCH --job-name=create_tars          # Job name
#SBATCH --partition=prepost                    # Take a node from the 'gpu' partition
#SBATCH --export=ALL                       # Export your environment to the compute node
##SBATCH --nodes=1            # Request N GPUs per machine

#SBATCH --cpus-per-task=1

#SBATCH --ntasks-per-node=1
##SBATCH --ntasks=1
#SBATCH --time=20:00:00

#SBATCH --array=0-31%16
#SBATCH --output=logs/slurm-%j-%a-archival.out 

#module load arch/a100


export PYTHONFAULTHANDLER=1

module load sox/14.4.2

source .venv/bin/activate


srun uv run archive_samples.py