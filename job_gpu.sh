#!/bin/sh
#SBATCH --job-name=exp1 # Job name
#SBATCH --ntasks=8 # Run on a eight CPU

#SBATCH --output=logs/batch_bert_exp1_%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G

# source /home/eshwarsr/IISc-DL-Project-3/virtual_env/bin/activate

conda activate base

python -u train_bert_classifier.py  >logs/redirect_bert_exp1_`date +%H_%M_%S_%d_%m_%Y`.log 2>&1

echo "Done"
#  ===== #SBATCH --time=10:00:00 # Time limit hrs:min:sec
