"""
This script has inline explicit code written in C to perform scalar
multiplication with 2 on floating point array using ElementWiseKernel.
"""
from time import time
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

DATA_SIZE = 10000000
host_data = np.float32(np.random.random(DATA_SIZE))

gpu_scalarmul_ker = ElementwiseKernel(
    arguments="float *in, float *out",  # arguments for the kernel function.
    operation="out[i] = 2 * in[i];",  # operation; cuda automatically fills i.
    name="gpu_scalarmul_ker",  # kernel function name.
)


def scalar_multiplication():
    # scalar multiplication on host.
    start_time = time()
    result_host = host_data * np.float32(2)
    end_time = time()
    run_time_hostop = end_time - start_time

    # scalar multiplication on device.
    device_data = gpuarray.to_gpu(host_data)
    result_device = gpuarray.empty_like(device_data)

    start_time = time()
    gpu_scalarmul_ker(device_data, result_device)
    end_time = time()
    run_time_deviceop = end_time - start_time

    # printing results for comparison of runtime between host and device.
    print(f"Run time for operation in host: {run_time_hostop:.6f} seconds.")
    print(f"Run time for operation in device: {run_time_deviceop: .6f} "
          f"seconds.")


if __name__ == "__main__":
    for run_number in range(5):
        scalar_multiplication()
