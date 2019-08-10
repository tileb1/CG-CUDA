#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --reservation=phpc2019gpu

source /ssoft/spack/bin/slmodules.sh -s  x86_E5v2_Mellanox_GPU

module load gcc cuda

make
srun ./run
