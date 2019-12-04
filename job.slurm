#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate tf2-gpu

srun python mnist2_classify.py
