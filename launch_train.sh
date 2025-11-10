#!/bin/bash
#SBATCH --job-name=pretrain_hubert          # Job name
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --hint=nomultithread 
#SBATCH --output=logs/slurm-%j-%t-pretrain.out 

##SBATCH --partition=prepost                    # Take a node from the 'gpu' partition

#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=8           # Request N GPUs per machine
#SBATCH --ntasks-per-node=4

## for H100 
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
module load arch/h100


export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load sox/14.4.2
source .venv/bin/activate

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export NCCL_P2P_DISABLE=1
#NCCL_P2P_DISABLE=1



nvidia-smi

srun uv run train.py --dataset longforms --dataset-path ./exp_iter2_B175/data/baby-hubert-175s_1_7 --exp-dir ./exp_iter3_B175 --feature-type hubert --num-class 500 --max-updates 400000 --seconds-per-batch 175 --learning-rate 0.0005 --gpus 4 --num-nodes 8