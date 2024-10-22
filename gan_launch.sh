#!/bin/bash -l
#SBATCH --job-name=composite
#SBATCH --output=/home/nas-wks01/users/uapv2300011/gan/results/log/simple_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=k2samy@hotmail.fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --partition=gpu

# Run your Python script
python main.py