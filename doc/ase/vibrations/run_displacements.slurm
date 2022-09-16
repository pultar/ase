#! /bin/bash -l
#SBATCH --job-name ethanol-disp
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=0:20:00
#SBATCH --output=array_%A-%a.out
#SBATCH --array=1-108

module purge
module load espresso

python3 displacement_db_run_file.py
