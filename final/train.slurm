#!/bin/bash
#SBATCH -o 0eecs600final.o%j
#SBATCH --time=1:00:00
#SBATCH -n 2 -N 1
#SBATCH --mem=2gb
#SBATCH --gres=gpu:1 -p gpufermi

# load needed software
module load caffe/2015

# copy in needed files
cp *.py depth_data.pickle2 "$PFSDIR"

# run code
"$PFSDIR"/final.py

# copy everything that was updated back
cp -ru "$PFSDIR"/* "$SLURM_SUBMIT_DIR"

# clean up after ourselves
rm -rf "$PFSDIR"/*

echo Finished.

