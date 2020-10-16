import tensorflow as tf

# compute nodes do not have internet access so download on head node

tf.keras.datasets.mnist.load_data()
