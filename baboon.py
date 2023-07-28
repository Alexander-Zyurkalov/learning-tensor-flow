from PIL import Image
import tensorflow as tf
import numpy as np

img = Image.open('baboon.jpg')
x_rgb = np.array(img).astype(np.float32)
x_rgb = tf.constant(x_rgb)

weights = tf.constant([[0.299], [0.587], [0.114]], dtype=tf.float32)

x = tf.matmul(x_rgb, weights)
x = tf.squeeze(x)

y = tf.constant(x)

y_reshaped = tf.reshape(y, [1, 308, 307, 1])

x = tf.nn.avg_pool(y_reshaped, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')


# Convert the tensor to a numpy array
x_np = tf.reshape(x, x.shape[1:3]).numpy()

# Normalize the array to 0-255
x_np = (x_np - np.min(x_np)) / (np.max(x_np) - np.min(x_np)) * 255

# Convert to uint8
x_np = x_np.astype(np.uint8)

# Create an image from the array
image = Image.fromarray(x_np)

# Save the image
image.save('output_smaller.jpg')
