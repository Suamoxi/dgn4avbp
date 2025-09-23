#!/bin/bash -l
#SBATCH --partition grace
#SBATCH -J train_nn_ae
#SBATCH --gres=gpu:1
#SBATCH --error %x%j.err
#SBATCH --output %x%j.out


#  with your own virtual environment, replace path to your path
singularity run --nv -B /home -B /scratch/ /softs/local_arm/singularity/images/pyg25.03.sif /scratch/coop/theret/avbpML_env/bin/python3 train_dgnavbp.py
