#!/bin/sh
#SBATCH --job-name=lr_exp6 # Job name
#SBATCH --ntasks=32 # Run on a eight CPU

#SBATCH --output=logs/batch_lr_exp6_%j.out # Standard output and error log
#SBATCH --partition=cl2_all_8G

source /home/eshwarsr/IISc-DL-Project-3/virtual_env/bin/activate

python -u train_lr.py >logs/redirect_lr_exp6_`date +%H_%M_%S_%d_%m_%Y`.log 2>&1

echo "Done"
#  ===== #SBATCH --time=10:00:00 # Time limit hrs:min:sec
