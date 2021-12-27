import tensorflow as tf

print("Checking for GPU in list of system devices:")
print("Num of GPUs available: ", len(tf.test.gpu_device_name()))