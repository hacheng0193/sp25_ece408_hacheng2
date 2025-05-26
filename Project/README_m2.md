# Milestone 2: Profiling Convolution and Implementing Kernel Fusion
***Deadline: April 11th, 2025 8PM***

In this second milestone, you will build upon the **basic CPU convolution, GPU convolution, and matrix unrolling** you completed in Milestone 1. Your tasks now include **profiling** those implementations and introducing a **kernel fusion** optimization.

Before starting Milestone 2, please watch the recorded profiling lecture available on the Canvas syllabus page. Although the lecture is scheduled for Week 10, you can view it at any time.

## Table of Contents
- [Overview and Deliverables](#overview-and-deliverables)
- [Setup](#setup)
- [1. Profiling Basic CPU Convolution](#1-profiling-basic-cpu-convolution)
- [2. Profiling Basic GPU Convolution and Matrix Unrolling](#2-profiling-basic-gpu-convolution-and-matrix-unrolling)
- [3. Implementing Kernel Fusion](#3-implementing-kernel-fusion)
- [4. Submitting Milestone 2 for Grading](#4-submitting-milestone-2-for-grading)
- [Rubric](#rubric)
- [Appendix](#appendix)



## Overview and Deliverables

For Milestone 2, you will:

1. **Profile** the CPU convolution using gprof.
2. **Profile** the GPU convolution and matrix unrolling using Nsight tools (nsys and ncu).
3. **Implement a new Kernel Fusion** approach, which fuses unrolling, matrix multiplication, and result permutation into one kernel.
4. **Profile** this fused kernel.

### Deliverables
| Deliverable                    | Description                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------- |
| **1. Implement Kernel Fusion** | Fuse unrolling + matrix multiplication + permutation into a single GPU kernel.   |
| **2. Complete the report**     | Complete a quiz-style report on PrairieLearn using your profiling results        |
| **3. Submit code for grading** | See [Submitting Milestone 2 for Grading](#4-submitting-milestone-2-for-grading). |

You will edit the following file for milestone 2.
```
project/src/layer/custom/kernel-fusion-forward.cu
```

**Only modify the files specifically mentioned in this document. Changes to other files will not be used for grading, and may cause unexpected errors that you will be responsible for.**



## Setup

1. **Pull the latest project updates** (if any) into your local repository.  
2. To compile, run:  
   ```bash
   ./run.sh build
   ```
   This will compile everything, including a binary (e.g., `m2`) for this milestone.
3. To execute your code, run (or edit the `.slurm` script to run):
   ```bash
   sbatch m2.slurm
   ```
   The `m2` binary is intended for kernel fusion.
4. To clean, run:
   ```bash
   ./run.sh clean
   ```
   This removes all compiled artifacts.

**Important:** If you are on a cluster (like Delta), use the appropriate Slurm commands (`srun`, `sbatch`) rather than running locally.



## 1. Profiling Basic CPU Convolution

In Milestone 1, you wrote a CPU implementation in a file similar to `cpu-new-forward.cc` (the function `conv_forward_cpu`). Now, you will **profile** that CPU version.

1. Compile with `-pg` flag in `run.sh`. Edit the following line.

    Original:
    ```bash
    cmake ./project/ && make -j8
    ```

    Updated:
    ```bash
    cmake -DCMAKE_CXX_FLAGS=-pg ./project/ && make -j8
    ```
2. Use Gprof to profile your CPU implementation for batch size of 1k.

    You will use `gprof` to profile the execution of your CPU forward convolution implementation.

    Compiling and linking your `cpu-new-forward.cc` with the `-pg` flag in the file `run.sh` will create a `gmon.out` artifact containing profile information when the binary `m1_cpu` is executed.  To analyze this information in human readable form, modify `m1_cpu.slurm` and modify the line to redirect `gprof` output as `outfile`.

        srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > outfile

    By default, `gprof` prints both a flat profile and a call graph (see "Interpreting gprof's Output" in the [GNU gprof Documentation](https://sourceware.org/binutils/docs/gprof/index.html)).  With the `-Q` flag, we only print the flat profile.  The information you need can be found near the beginning of `gprof`'s output. You can download your build folder and process the output `outfile` with `grep` (with your function's name) or `head`. You can also open it with a text editor if you want to examine the complete output.

3. Remove `-pg` flag in `run.sh` when you finish CPU profiling. It will slow down your program significantly.


## 2. Profiling Basic GPU Convolution and Matrix Unrolling

In Milestone 1, you created GPU convolution kernel and matrix unrolling kernel. Now it's time to collect in-depth performance information.

The following instructions will use `m1_gpu` as an example to demonstrate the profiling process. Matrix unroll should be profiled in the same way.

You will use the profiling results to complete the report.

### Using Nsight-Systems and Nsight-Compute

**Before you do any profiling, make sure your implementation achieves desired accuracy. Also make sure you do not have any memory errors by running `compute-sanitizer`. See [Checking for Errors](#appendix) on how to run this.**

***System level profiling using Nsight-Systems***

We will learn how to use `nsys` (Nsight Systems) to profile the execution at the application level.

Once you've gotten the appropriate accuracy results, generate a profile using `nsys`.
You have to remove `-DCMAKE_CXX_FLAGS=-pg` in `run.sh` and make line of your `run.sh`:

    cmake ./project/ && make -j8

Then, modify `m1_gpu.slurm` to generate a profile instead of just executing the code. The output is inside `profile.out` file.

    srun nsys profile --stats=true ./m1_gpu > profile.out

You should see something that looks like the following (but not identical):

```bash 
......

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)          Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  -----------  ---------------------
     99.9  351,122,724,860      3,519  99,779,120.4  100,089,303.0     2,855  100,130,281  5,413,528.2  poll                 
      0.1      283,382,530        925     306,359.5       14,207.0     1,051   20,208,549  1,050,067.9  ioctl                
     ......               
      0.0            1,913          1       1,913.0        1,913.0     1,913        1,913          0.0  bind                 

[5/8] Executing 'cudaapisum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     ......     

[6/8] Executing 'gpukernsum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      GridXYZ         BlockXYZ                                               Name                                          
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ---------------  --------------  ----------------------------------------------------------------------------------------
     ......                                                                   

[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)       Operation     
 --------  ---------------  -----  -------------  -------------  -----------  -----------  ------------  ------------------
     ......

[8/8] Executing 'gpumemsizesum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)   StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  ---------  -----------  ------------------
     ......

```

The CUDA API Statistics section shows the CUDA API calls that are executed. The CUDA Kernel Statistics lists all the kernels that were executed during the profiling session. There are also more details on the CUDA memory operations (CudaMemcpy) listed.
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls. Use **your** `nsys` profiling output corresponding to the section above to answer the questions for your quiz.

You can find more information about `nsys` in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling)

***Kernel level profiling using Nsight-Compute***

When doing profiling tasks with Nsight-Compute, modify your SLURM configuration as follows:

```bash
#SBATCH --constraint="projects,perf,nvperf"
```

For regular development and debugging, use the standard configuration:

```bash
#SBATCH --constraint="projects"
```

**Note:** Only use the profiling configuration when actively collecting performance metrics. The `perf,nvperf` constraints instruct Delta to reserve a dedicated 4×A40 node for your job. This ensures that no other processes interfere with profiling but may result in longer wait times for node allocation. Using the standard configuration during development helps maintain resource availability for other cluster users by avoiding unnecessary exclusive node allocation.

Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `ncu` (Nsight-Compute).

1. Modify `m1_gpu.slurm` to use `ncu` to save some timeline and analysis information.
   ```bash
   srun ncu --set full -f -o analysis_file ./m1_gpu 100 > gpu_ncu.out
   ```
   This generates `analysis_file.ncu-rep`.
2. Download that `.ncu-rep` file locally to open in the **Nsight Compute GUI**.  
3. Examine memory behavior, SM efficiency, etc. to find performance bottlenecks.

You can find more information about `ncu` in the [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

## 3. Implementing Kernel Fusion

Modify `./project/src/layer/custom/kernel-fusion-forward.cu` to create the kernel fusion implementation of input unrolling.

**Kernel Fusion** fuses the matrix unrolling kernel, the matrix multiplication kernel, and the permutation kernel into **one** kernel. This technique is covered as "Matrix-Multiplication with built-in unrolling" in the lecture. Below is a more detailed explanation of this technique.

The implementation starts with the tiled matrix multiplication kernel. You can refer to your lab3 code or `./project/src/layer/custom/matmul.cu`.
- When loading input elements into shared memory, instead of reading from a pre-unrolled matrix, the kernel directly loads the corresponding elements from the original input feature.
- After computing the output element, the kernel writes it to global memory. At this stage, it directly stores the results in the correct positions, applying the necessary permutation.

The skeleton code is provided, and the places you need to complete are marked with `TODO`.

**Profile** your fused kernel using Nsight Systems/Compute again. Compare the time consumed by the fused version vs. your separate-kernel approach. You will use the profiling results to complete the report.

The sum of Op times on batch_size=10000 should be approximately 60 ms if you implement the fused kernel correctly. To earn full credits on the coding part, you must
- have correct accuracies for any batch size
- achieve total Op time less than 200 ms for batch_size=10000
- use tiled matrix multiplication

We will measure Op times without profiling. When verifying performance, you can use results from non-profiling runs or `nsys` profiling, as `ncu` profiling introduces significant overhead.


## 4. Submitting Milestone 2 for Grading

To submit your work for grading, add, commit, and push your files:

* ```git add -u```
* ```git commit -m "some comment"```
* ```git push origin main```

Do not add profiling results (`.sqlite`, `.nsys-rep`, `.ncu-rep`) to Git. These files are not required for code grading and are often large, potentially exceeding GitHub’s size limit. Including them may prevent you from successfully pushing your commit.  

Make sure to complete your quiz on PrairieLearn. Double check you finish all items listed in the Deliverables for this milestone.

The code in your GitHub repository at 8pm on Monday, April 14 will be considered the final version, even if you had a correct version earlier. We will retrieve and grade the code at that time.

## Rubric

| Component | Percentage |
| --------- | ---------: |
| **Code**  |        15% |
| **Quiz**  |        15% |


## Appendix

### Checking for Errors

In Milestone 1, we discussed using `cuda-memcheck` to detect memory errors. However, students have reported that this tool is now considered deprecated, and `compute-sanitizer` is the recommended replacement.

To check for memory errors in the Milestone 1 GPU binary, update your m1_gpu.slurm script to include the following command:
```
srun compute-sanitizer m1_gpu 100 > m1_gpu.out
```


