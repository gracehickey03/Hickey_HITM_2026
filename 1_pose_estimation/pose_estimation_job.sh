#!/bin/bash

#SBATCH --job-name=video_analysis   	# Job name 
#SBATCH --mail-type=ALL			        # Mail events 
#SBATCH --mail-user=gch22a@fsu.edu     	# Where to send mail 
#SBATCH --mem=32gb                     	# Job memory request 
#SBATCH --time=12:00:00               	# Time limit hrs:min:sec 
#SBATCH --output=video_analysis_%j.log  # Standard output and error log 
#SBATCH --nodes=1
#SBATCH --gpus=1  
#SBATCH --partition=hpg-b200 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
pwd; hostname; date 

module purge
module load cuda/12.8.1
module load conda

conda activate DEEPLABCUT2

python /blue/adamdewan/gch22a.fsu/DLC/Scripts/pose_estimation.py # path to pose estimation python script
