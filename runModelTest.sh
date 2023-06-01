#!/bin/bash  

# The --job-name is the name of the job as it appears in the queue.
#SBATCH --job-name=deepOtecMonth

# The --output parameter specifies the file where standard errors from your job
# will be written (The %j in the name gets replaced by the Slurm job number).
#SBATCH --error=JobID.%j.error

# The Queue that will be used
#SBATCH --account=dwarsing

# The --mem parameter specifies the required memory on the node that is going to be used, it is specified in Mb.
#SBATCH --mem=10240MB
# Specify the time you expect the job to run.  For 10 hours, set the time to 10:00:00.
#SBATCH --time=04:30:00
#Specify the number of GPUs being used per node, we only have one GPU
#SBATCH --gpus-per-node=1

# Specify the e-mail address you would like job status e-mails to be sent to.
#SBATCH --mail-user=adshowal@purdue.edu

# Specify the types of job status messages you would like to receive e-mail about:
#  BEGIN - send a message when the job starts.
#  FAIL - send a messages if the job fails.
#  END - send a message when the job runs to completion.
#SBATCH --mail-type=BEGIN,FAIL,END

# At this point, all of the job parameters for Slurm have been set by the SBATCH lines above.
# Everything that follows are the commands you want this job to execute.

module load anaconda
module load cuda
conda activate python3.8-torch

python test.py

