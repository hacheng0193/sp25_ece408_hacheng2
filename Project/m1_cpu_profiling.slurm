#!/bin/bash
#SBATCH --job-name="Milestone1_CPU"
#SBATCH --output="Milestone1_CPU.out"
#SBATCH --error="Milestone1_CPU.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m1_cpu 100 > m1_cpu.out

# Generate running time information
#{ time srun ./m1_cpu 100 > m1_cpu.out; } 2> time.out

# Profiling Basic CPU Convolution
# srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > outfile

# Basic GPU NSys Profiling
#srun nsys profile --stats=true ./m1_gpu > profile.out

srun ncu --set full -f -o analysis_file ./m1_gpu 10000 > gpu_ncu.out