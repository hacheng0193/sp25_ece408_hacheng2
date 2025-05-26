#!/bin/bash
#SBATCH --job-name="Milestone3_0"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
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

# srun ncu --metrics sm__pipe_tensor_active.avg.pct_of_peak_sustained_active ./m3 &> tensor_usage.txt


# Generate nsys profile information, output will be in profile.out and report.nsys-rep
srun nsys profile --stats=true ./m3 > m3_nsys_profile.out

# Generate ncu profile information, output will be in analysis_file.ncu-rep
srun ncu --set full -f -o analysis_file ./m3 > m3_ncu.out