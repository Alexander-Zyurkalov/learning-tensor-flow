import numpy as np
import tensorflow as tf

signal = tf.reshape(tf.constant(np.array([1, 2, 3, 4, 5, 6]).astype("float32")), [1, 6, 1])
filter = tf.reshape(tf.constant(np.array([0, 1]).astype("float32")), [2, 1, 1])

c = tf.nn.convolution(signal, filter)
print(c)
