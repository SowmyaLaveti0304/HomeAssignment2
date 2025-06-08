import numpy as np
import tensorflow as tf

# Define the 5x5 input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# Reshape input: (batch, height, width, channels)
input_tensor = input_matrix.reshape((1, 5, 5, 1))

# Define the 3x3 kernel
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

# Reshape kernel: (height, width, in_channels, out_channels)
kernel_tensor = kernel.reshape((3, 3, 1, 1))

# Function to perform convolution
def perform_convolution(input_tensor, kernel_tensor, stride, padding):
    return tf.nn.conv2d(input_tensor, kernel_tensor, strides=[1, stride, stride, 1], padding=padding)

# Perform convolutions with different parameters
configs = {
    "Stride=1, Padding='VALID'": perform_convolution(input_tensor, kernel_tensor, stride=1, padding='VALID'),
    "Stride=1, Padding='SAME'": perform_convolution(input_tensor, kernel_tensor, stride=1, padding='SAME'),
    "Stride=2, Padding='VALID'": perform_convolution(input_tensor, kernel_tensor, stride=2, padding='VALID'),
    "Stride=2, Padding='SAME'": perform_convolution(input_tensor, kernel_tensor, stride=2, padding='SAME')
}

# Print results
for config, result in configs.items():
    print(f"\n{config}:\n{result.numpy().squeeze()}")
