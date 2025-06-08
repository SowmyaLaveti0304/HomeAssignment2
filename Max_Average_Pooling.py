import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

### --- Task 1: Edge Detection using Sobel Filters --- ###

# Load and convert to grayscale
image = cv2.imread('/Users/sowmyalaveti/Documents/Summer 2025/Neural networks/dog-8198719_1280.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded properly
if image is None:
    raise FileNotFoundError("Image file not found. Please ensure image exists.")

# Define Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# Apply Sobel filters
edges_x = cv2.filter2D(image, -1, sobel_x)
edges_y = cv2.filter2D(image, -1, sobel_y)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Sobel X")
plt.imshow(edges_x, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Sobel Y")
plt.imshow(edges_y, cmap='gray')

plt.tight_layout()
plt.show()

### --- Task 2: Max Pooling and Average Pooling --- ###

# Random 4x4 matrix as input
input_matrix = np.random.randint(0, 10, size=(1, 4, 4, 1)).astype(np.float32)
print("Original 4x4 Matrix:\n", input_matrix[0, :, :, 0])

# Max Pooling
max_pooled = tf.nn.max_pool2d(input_matrix, ksize=2, strides=2, padding='VALID')
print("\nMax Pooled 2x2:\n", max_pooled.numpy()[0, :, :, 0])

# Average Pooling
avg_pooled = tf.nn.avg_pool2d(input_matrix, ksize=2, strides=2, padding='VALID')
print("\nAverage Pooled 2x2:\n", avg_pooled.numpy()[0, :, :, 0])
