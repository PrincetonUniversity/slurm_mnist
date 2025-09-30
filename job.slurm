#!/bin/bash
#SBATCH --job-name=tf-test          # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                    # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1                # number of gpus per node
#SBATCH --time=00:05:00             # total run time limit (HH:MM:SS) 
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2025.6
conda activate tf-gpu

python mnist_classify.py
