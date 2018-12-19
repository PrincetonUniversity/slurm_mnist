#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=yournetid @princeton.edu

module load anaconda3
source activate tf
python mnist_classify.py
