#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ld2494/%j_%x.out
#SBATCH --error=/scratch/ld2494/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-user=<ld2494@nyu.edu>
#SBATCH --mail-type=ALL

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --bind /scratch/ld2494 --overlay /scratch/ld2494/overlay-25GB-500K.ext3:ro /scratch/ld2494/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/ld2494/run_hyperparameter_search.py BoolQ
"
