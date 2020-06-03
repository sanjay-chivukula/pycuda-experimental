"""
This script is written to play around with overloaded arithmetic operators
used to perform operations on device data and compare result types with host
operator results.
"""
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np

# Revising old concepts.
gpu_device_count = cuda.Device.count()
print(f"Device Count: {gpu_device_count}")

try:
    gpu_device = cuda.Device(0)
    print(f"Device Name: {gpu_device.name()}")
except cuda.LogicError as l_error:
    print("Check the ordinal used to access the cuda device.")
    raise l_error

# host data.
# remember, it is better to explicitly set numpy array dtype.
x_host = np.array(
    [1, 2, 3],
    dtype=np.float32  # Equivalent to C_type: float.
)

y_host = np.array(
    [2, 3, 5],
    dtype=np.float64  # Equivalent to C_type: double.
)


# device data.
x_device = gpuarray.to_gpu(x_host)
y_device = gpuarray.to_gpu(y_host)
result_device = x_device ** y_device
result_host = result_device.get()

print(result_host, result_host.dtype)