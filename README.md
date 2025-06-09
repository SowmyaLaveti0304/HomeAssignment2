# Home Assignment 2 – Summer 2025  
Student Name: Sowmya Laveti  
Student ID: 700771347  
University: University of Central Missouri  
Course: CS5720 – Neural Networks and Deep Learning  

---

## Assignment Overview  
This assignment is divided into three key parts:

1. Convolution Operations with Different Parameters
2. CNN Feature Extraction with Filters and Pooling
3. Data Preprocessing - Standardization vs. Normalization

---
## Part 1: Convolution Operations with Different Parameters
### Tasks Completed:

- Defined the **5×5 input matrix** and applied convolution using the **3×3 kernel** with different stride and padding configurations:
  - **Input Matrix**:
    ```plaintext
    [[ 1,  2,  3,  4,  5],
     [ 6,  7,  8,  9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20],
     [21, 22, 23, 24, 25]]
    ```
  - **Kernel**:
    ```plaintext
    [[ 0,  1,  0],
     [ 1, -4,  1],
     [ 0,  1,  0]]
    ```
  - **Convolution configurations used**:
    - Stride = 1, Padding = 'VALID'
    - Stride = 1, Padding = 'SAME'
    - Stride = 2, Padding = 'VALID'
    - Stride = 2, Padding = 'SAME'

Each configuration was implemented using TensorFlow, and the resulting feature maps were printed to analyze the effect of stride and padding on the output size and values.

---

### Outputs:

- **Stride = 1, Padding = 'VALID'**
    ```plaintext
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    ```

- **Stride = 1, Padding = 'SAME'**
    ```plaintext
    [[  4.   3.   2.   1.  -6.]
     [ -5.   0.   0.   0. -11.]
     [-10.   0.   0.   0. -16.]
     [-15.   0.   0.   0. -21.]
     [-46. -27. -28. -29. -56.]]
    ```

- **Stride = 2, Padding = 'VALID'**
    ```plaintext
    [[0. 0.]
     [0. 0.]]
    ```

- **Stride = 2, Padding = 'SAME'**
    ```plaintext
    [[  4.   2.  -6.]
     [-10.   0. -16.]
     [-46. -28. -56.]]
    ```
---
### Summary:

This task helps us understand how changes in **stride** and **padding** affect the output of a convolution operation.

- Using SAME padding keeps the output size close to the original input, helping preserve details—especially around the edges.

- Increasing the stride skips more pixels, which results in a smaller output—similar to zooming out or compressing the image.

## Part 2: CNN Feature Extraction with Filters and Pooling

### Task 1: Edge Detection Using Sobel Filter

- Loaded a grayscale image.
- Applied the **Sobel filter** for edge detection in both **X-direction** and **Y-direction**.
- Displayed the **original image**, the **Sobel-X output**, and the **Sobel-Y output** for visual comparison.
- **Sobel filters used**:

  - Sobel-X (detects vertical edges):
    ```plaintext
    [-1  0  1]
    [-2  0  2]
    [-1  0  1]
    ```

  - Sobel-Y (detects horizontal edges):
    ```plaintext
    [-1 -2 -1]
    [ 0  0  0]
    [ 1  2  1]
    ```

---

### Task 2: Max Pooling and Average Pooling

- Created a **random 4×4 matrix** to simulate an input image.
- Demonstrated both **2×2 Max Pooling** and **2×2 Average Pooling** using TensorFlow.
- Printed the:
  - Original matrix  
  - Max-pooled matrix  
  - Average-pooled matrix  

This task shows how pooling helps reduce spatial dimensions while retaining the essential features of the input.
### Summary

This part of the assignment demonstrates how **filters and pooling** work in convolutional neural networks:

- **Sobel filters** help detect **edges** in images by identifying sharp changes in pixel intensity.  
  - **Sobel-X** highlights vertical edges.
  - **Sobel-Y** highlights horizontal edges.

- **Pooling operations** are used to **reduce the size** of feature maps while preserving important information.  
  - **Max pooling** captures the most prominent features by selecting the maximum value in each region.
  - **Average pooling** provides a smoother result by taking the mean of each region.

Together, these techniques form the foundation for how CNNs extract and simplify features from images.

## Part 3: Data Preprocessing – Standardization vs. Normalization

### Task Overview

This task focuses on preparing data for machine learning models through preprocessing techniques.

- **Dataset Used:**  
  Iris dataset from `sklearn.datasets`

- **Transformations Applied:**
  - **Min-Max Normalization** using `MinMaxScaler`
  - **Z-score Standardization** using `StandardScaler`

- **Visualization:**  
  Histograms were plotted to compare the distributions before and after transformation for each feature.

- **Model Training:**  
  A **Logistic Regression** model was trained on:
  - Raw (unprocessed) data  
  - Normalized data  
  - Standardized data  

- **Objective:**  
  To analyze the effect of preprocessing on model performance and understand when to use **normalization** versus **standardization**.

---

### Summary

- **Normalization** scales features to a fixed range (typically [0, 1]) and is useful when input values vary widely or when using models sensitive to feature scales (e.g., neural networks, image data).
- **Standardization** centers features around the mean with unit variance, which is ideal for data following a normal distribution and for gradient-based models like logistic regression and deep learning.

Proper preprocessing improves model accuracy, stability, and convergence speed.

## ▶ How to Run

### 1. Install Dependencies
Before running the scripts, make sure you have all required libraries installed:
```bash
pip install -r requirements.txt
```
### Run Scripts
python Convolution_Operations.py
python Edge_Detection.py
python Max_Average_Pooling.py
python DataPreprocessing.py



 
  



