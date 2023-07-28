#!/bin/bash  

#SBATCH --job-name=deepOtecMonth
#SBATCH --error=JobID.%j.error
#SBATCH --account=dwarsing-h
#SBATCH --mem=10240MB
#SBATCH --time=04:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=lied@purdue.edu
#SBATCH --mail-type=BEGIN,FAIL,END

module load anaconda/2020.11-py38
module load cuda/11.2.0

python OTEC.py