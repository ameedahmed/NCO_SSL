#!/usr/bin/env bash
#SBATCH --job-name=KNN_TEST
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=ahmed003@ismll.de
#SBATCH --partition=STUDL
#SBATCH --gres=gpu:1


cd /home/ahmed003/NCO_code_KNN/single_objective/LEHD/TSP/test.py # navigate to the directory if necessary


srun /home/ahmed003/miniconda3/envs/srpenv/bin/python3.10 test.py     # run the script