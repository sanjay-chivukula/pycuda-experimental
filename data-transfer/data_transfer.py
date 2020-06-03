"""
This is a sample code to transfer and work with data between cpu and gpu using
gpuarray.
Reference: Hands-On GPU Programming with Python and CUDA (Book).
"""

import numpy as np

import pycuda.autoinit  # Required: to use pycuda.
from pycuda import gpuarray
import sys

# Setting up host data.
sample = [1, 2, 3, 4, 5]
host_data = np.array(
    sample,
    dtype=np.float32,
)

# Transferring host data to device.
device_data = gpuarray.to_gpu(host_data)

# Processing data in device.
device_data = 2 * device_data

# Transferring processed device data to host.
host_data = device_data.get()

print(f"Processed host data: \n{host_data}")