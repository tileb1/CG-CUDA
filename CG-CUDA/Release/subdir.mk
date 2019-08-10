################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

C_SRCS += \
../helper.c \
../sequential.c 

OBJS += \
./helper.o \
./main.o \
./sequential.o 

CU_DEPS += \
./main.d 

C_DEPS += \
./helper.d \
./sequential.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/ssoft/spack/paien/v2/opt/spack/linux-rhel7-x86_E5v2_Mellanox_GPU/gcc-6.4.0/cuda-9.1.85-qrb6dpbuhogia6oonjhj7du6afxdnkvr/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/ssoft/spack/paien/v2/opt/spack/linux-rhel7-x86_E5v2_Mellanox_GPU/gcc-6.4.0/cuda-9.1.85-qrb6dpbuhogia6oonjhj7du6afxdnkvr/bin/nvcc -O3 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/ssoft/spack/paien/v2/opt/spack/linux-rhel7-x86_E5v2_Mellanox_GPU/gcc-6.4.0/cuda-9.1.85-qrb6dpbuhogia6oonjhj7du6afxdnkvr/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/ssoft/spack/paien/v2/opt/spack/linux-rhel7-x86_E5v2_Mellanox_GPU/gcc-6.4.0/cuda-9.1.85-qrb6dpbuhogia6oonjhj7du6afxdnkvr/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


