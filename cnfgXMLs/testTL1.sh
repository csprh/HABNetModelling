#! /bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --mem=16G

module add languages/anaconda3/3.5-4.2.0-tflow-1.7
#module add languages/anaconda2/5.0.1.tensorflow-1.6.0

#which python
#onda list
cd /mnt/storage/home/csprh/code/HAB/classifyHAB1
python trainHAB.py classifyHAB1.xml 
