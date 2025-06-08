import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load a grayscale image (you can replace 'image.jpg' with any valid path)
image = cv2.imread('/Users/sowmyalaveti/Documents/Summer 2025/Neural networks/dog-8198719_1280.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded properly
if image is None:
    raise FileNotFoundError("Image file not found. Please ensure image exists.")

# Define Sobel filters
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Apply Sobel filter using cv2.filter2D
edges_x = cv2.filter2D(image, -1, sobel_x)
edges_y = cv2.filter2D(image, -1, sobel_y)

# Combine edges using magnitude
edges_combined = cv2.magnitude(np.float32(edges_x), np.float32(edges_y))

# Display the original and edge-detected images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Sobel - X direction')
plt.imshow(edges_x, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Sobel - Y direction')
plt.imshow(edges_y, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Combined Magnitude')
plt.imshow(edges_combined, cmap='gray')

plt.tight_layout()
plt.show()
