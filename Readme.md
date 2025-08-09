# Minimal CNN from Scratch in C++

This repository contains a minimal, bare-bones implementation of a Convolutional Neural Network (CNN) written purely in C++ without any external ML libraries.  
It demonstrates the core concepts of CNNs using simple math and data structures for educational purposes.

---

## Code Explanation

### 1. Activation Functions

```cpp
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double dsigmoid(double y) { return y * (1 - y); }
```

````

- **`sigmoid`**: Squashes input values to the range (0, 1).
- **`dsigmoid`**: Derivative of sigmoid, used for backpropagation weight updates.

### 2. Convolution Operation (`conv2d`)

```cpp
vector<vector<double>> conv2d(const vector<vector<double>> &img,
                              const vector<vector<double>> &kernel);
```

- Performs 2D convolution by sliding the kernel (filter) over the image, computing the sum of element-wise products.
- Produces a smaller output called the feature map.

### 3. Flatten Function (`flatten`)

Converts the 2D feature map into a 1D vector so it can be fed into a fully connected layer.

### 4. Input Image and Kernel

- `img`: A 5×5 sample image represented as a 2D vector of pixel values (0 or 1).
- `kernel`: A 3×3 convolution filter initialized with random weights.

### 5. Fully Connected Layer Weights and Bias

- `fc_weights`: Weights connecting flattened features to the output neuron.
- `bias`: Bias term added before activation.
- `lr`: Learning rate controlling weight updates.

### 6. Forward Pass

- Convolve the input image with the kernel.
- Flatten the result.
- Calculate weighted sum plus bias.
- Apply sigmoid activation to get the output.

### 7. Error Calculation and Backpropagation

- Compute the error between the predicted output and target.
- Calculate the delta (gradient) for weight updates.

### 8. Update Weights and Bias

- Adjust fully connected layer weights and bias using the delta and learning rate.
- Update the convolution kernel weights in a simplified manner.

---

## How to Run

1. Compile:

```bash
g++ -O2 -o minimal_cnn minimal_cnn.cpp -std=c++11
```

2. Run:

```bash
./minimal_cnn
```

---

## Notes

- This example is for educational purposes only and not optimized for real-world use.
- Real CNNs involve multiple layers, different activations (ReLU), batch training, and hardware acceleration.
- Backpropagation to convolution kernels here is simplified for clarity.

---

## Next Steps

- Add more convolution filters and layers.
- Implement ReLU and pooling layers.
- Train on real datasets like MNIST.
- Use batch training and better optimizers like Adam.


Made by Pujan — Happy coding! 🚀



````
