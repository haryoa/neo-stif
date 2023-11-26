#!/bin/bash
#SBATCH --job-name="train-insertion-stif"
#SBATCH --output="../run_slurm_1.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH -p ws-ia*
#SBATCH --qos=gpu-8
#SBATCH --time=12:00:00

hostname
python -m neo_stif train_stif insertion koto 32 --with-validation --do-compute-class-weight --train-path data/scolid/train_with_pointing.csv --dev-path data/scolid/val_with_pointing.csv --processed-train-data-path data/scolid/train_insertion --processed-dev-data-path data/scolid/val_insertion --output-dir-path output/scolid-i-f/felix-insertion/