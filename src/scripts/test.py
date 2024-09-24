import cupy as cp

print(cp.cuda.runtime.getDeviceCount())  # Should return the number of available GPUs

print(cp.cuda.runtime.runtimeGetVersion())
