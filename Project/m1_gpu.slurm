#!/bin/bash
#SBATCH --job-name="Milestone1_GPU"
#SBATCH --output="Milestone1_GPU.out"
#SBATCH --error="Milestone1_GPU.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

#srun ./m1_gpu 10000 > m1_gpu.out
{ time srun ./m1_gpu 100 > m1_gpu_100.out; } 2> time_100.out
{ time srun ./m1_gpu 1000 > m1_gpu_1000.out; } 2> time_1000.out
{ time srun ./m1_gpu 10000 > m1_gpu_10000.out; } 2> time_10000.out