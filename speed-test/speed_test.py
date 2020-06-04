"""
This script is written to compare the speed of computation of an operation
between host and device.
"""
from time import time
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray

DATA_SIZE = 10000000
host_data = np.float32(np.random.random(DATA_SIZE))

for run_number in range(10):
    # scalar multiplication with 2 on host machine.
    start_time = time()
    result_host_operation = host_data * np.float32(2)
    end_time = time()
    run_time_hostop = end_time - start_time

    # scalar multiplication with 2 on device machine.
    device_data = gpuarray.to_gpu(host_data)

    start_time = time()
    result_device_operation = device_data * np.float32(2)
    end_time = time()
    run_time_deviceop = end_time - start_time

    # runtime display for comparison.
    print(f"For run number: {run_number}")
    print(f"Host machine took: {run_time_hostop:.4f} seconds")
    print(f"Device machine took: {run_time_deviceop: .4f} seconds")

