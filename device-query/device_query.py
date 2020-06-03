"""
This script displays information about each CUDA enabled device avaiable.
Reference: Hands-On GPU Programming with Python and CUDA (Book).
"""

import pycuda.driver as cuda

cuda.init()  # Required: to use pycuda.driver

# cuda_cores_per_mp = {compute_capability: cuda_cores_per_multiprocessor}
cuda_cores_per_mp = {
    5.0: 128,
    5.1: 128,
    5.2: 128,
    6.0: 64,
    6.1: 128,
    6.2: 128,
}

cuda_device_count = cuda.Device.count()
print(f"Detected {cuda_device_count} CUDA enabled devices. \n")

for device_idx in range(cuda_device_count):
    gpu_device = cuda.Device(device_idx)
    print(f"Device {device_idx} Name: {gpu_device.name()}")

    major, minor = gpu_device.compute_capability()
    compute_capability = float(f"{major}.{minor}")
    print(f"\t Compute Capability: {compute_capability}")

    # total_memory() returns value in bytes.
    total_memory_mb = gpu_device.total_memory() // 1024 ** 2
    print(f"\t Total Memory: {total_memory_mb} megabytes")

    # get_attributes() returns a dict containing the device information.
    device_attributes = gpu_device.get_attributes()
    for key, value in device_attributes.items():
        print(f"\t {key}: {value}")
