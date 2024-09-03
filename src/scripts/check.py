import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"Number of GPUs available: {len(gpus)}")

print(tf.__version__)
