#!/usr/bin/env bash
#SBATCH --job-name=CrSigmoidImpl
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=ahmed003@ismll.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd /home/ahmed003/NCO_code_CR/single_objective/LEHD/TSP/train.py # navigate to the directory if necessary


srun /home/ahmed003/miniconda3/envs/srpenv/bin/python3.10 train.py     # run the script