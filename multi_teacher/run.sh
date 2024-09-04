#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --job-name=emb_cifar10_train_multi_distil
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_logs/logs-%j.out
#SBATCH --error=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_errors/error-%j.err

pyfile=/home/mila/s/sparsha.mishra/projects/hypa-scaleup/multi_teacher/without_simclr.py

module load anaconda/3

conda activate /home/mila/s/sparsha.mishra/.conda/envs/sparse

ulimit -Sn $(ulimit -Hn)

python3 $pyfile --batch-size=256;
