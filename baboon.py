from PIL import Image
import tensorflow as tf
import numpy as np

x_rgb = np.array(Image.open('baboon.jpg')).astype(np.float32)
x_rgb = tf.constant(x_rgb)
print(x_rgb.shape)

grays = tf.constant([[0.299], [0.587], [0.114]], dtype=tf.float32)

x = tf.matmul(x_rgb, grays)
x = tf.squeeze(x)


# Convert the tensor to a numpy array
x_np = x.numpy()

# Normalize the array to 0-255
x_np = (x_np - np.min(x_np)) / (np.max(x_np) - np.min(x_np)) * 255

# Convert to uint8
x_np = x_np.astype(np.uint8)

# Create an image from the array
image = Image.fromarray(x_np)

# Save the image
image.save('output.jpg')