HOST_COMP ?= mpicxx  
ARCH ?= sm_60
NVCC = nvcc

TARGET = wave_cuda_mpi
SRC = mpi_cuda.cu

all:
        $(NVCC) -arch=$(ARCH) -O2 -std=c++11 -ccbin=$(HOST_COMP) $(SRC) -o $(TA$

clean:
        rm -f $(TARGET)

test:
        mpisubmit.pl -p 4 -g 4 --stdout c_4_4.out $(TARGET)
