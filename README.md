# PyCUDA Experiments:
This repository contains notes and code of my experiments and learning with
 PyCUDA.

## Leaning PyCUDA:
Am following "Hands-On GPU Programming with Python and CUDA" by Dr.&nbsp;Brian
&nbsp;Tuomanen for learning PyCUDA. These projects below are in sequence
 according to the exercises and samples from the book. 
  
 1. [Device Query](./device-query/readme.md)
 1. [Basic Data Transfer](./data-transfer/readme.md)
 1. [Basic Arithmetic Operations](./arithmetic-operations/readme.md)
 1. [Speed Test](./speed-test/readme.md)

**Note**:   
If you are using pycuda with pycharm(Ubuntu) you may get an error saying
 missing nvcc, even though you installed it properly and it works in terminal
 . You can fix this by adding the right PATH and LD_LIBRARY_PATH variables in
  project run and debug configurations.