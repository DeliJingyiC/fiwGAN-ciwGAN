#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=PAS1957
#SBATCH --gpus-per-node=1
#SBATCH --output=sbatch/%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=28

echo $SLURM_JOB_ID

module load miniconda3/4.10.3-py37
source activate
conda activate tsf2


#conda install tensorflow-gpu yapf tensorboard=2.6 tqdm -y
#conda install -c conda-forge librosa -y

conda env list
set -x

export TF_CPP_MIN_LOG_LEVEL=1

python train_fiwgan.py --mode=train --train_dir=model --data_dir=setTwo --train_batch_size=64 --data_pad_end --wavegan_loss='wgan-gp' --job_id=$SLURM_JOB_ID --data_slice_len=16384 --data_num_channels=1 --num_epoch=500 --num_categ=3 # > "sbatch/${SLURM_JOB_ID}_main.log"