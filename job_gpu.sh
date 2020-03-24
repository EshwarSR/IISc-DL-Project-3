#!/bin/sh
#SBATCH --job-name=exp5 # Job name
#SBATCH --ntasks=8 # Run on a eight CPU

#SBATCH --output=logs/batch_exp5_%j.out # Standard output and error log
#SBATCH --partition=cl1_all_4G

source /home/eshwarsr/IISc-DL-Project-3/virtual_env/bin/activate

python -u train_rnn.py "EXP5" 5 >logs/redirect_exp5_`date +%H_%M_%S_%d_%m_%Y`.log 2>&1

echo "Done"
#  ===== #SBATCH --time=10:00:00 # Time limit hrs:min:sec
