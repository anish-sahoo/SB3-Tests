#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=10:00:00
#SBATCH --job-name=stablebaselines3-1tb-large
#SBATCH --mem=1024GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

## <your code>
# Load required modules
module load conda
module load cuda/12.1

# Create a conda environment with Python 3.11
conda create --name stable_env python=3.11 -y

source activate stable_env

pip3 install stable-baselines3[extra]
conda install tensorflow tensorboard
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

cd /home/sahoo.an/Projects/SB3-Tests
python3 e.py
