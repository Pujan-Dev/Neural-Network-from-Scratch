
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

// Sigmoid activation
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double dsigmoid(double y) { return y * (1 - y); }

// Simple 2D convolution : its the heart of the CNN it takes args like img and
// kernel -> which is like an filter ;
vector<vector<double>> conv2d(const vector<vector<double>> &img,
                              const vector<vector<double>> &kernel) {
  int h = img.size(), w = img[0].size();
  int kh = kernel.size(), kw = kernel[0].size();
  vector<vector<double>> out(h - kh + 1, vector<double>(w - kw + 1));
  for (int i = 0; i <= h - kh; i++)
    for (int j = 0; j <= w - kw; j++) {
      double sum = 0;
      for (int ki = 0; ki < kh; ki++)
        for (int kj = 0; kj < kw; kj++)
          sum += img[i + ki][j + kj] * kernel[ki][kj];
      out[i][j] = sum;
    }
  return out;
}

// Flatten matrix to vector -> sorry but i have to flatten to vector (ref: flat
// earth)
vector<double> flatten(const vector<vector<double>> &m) {
  vector<double> v;
  for (auto &row : m)
    v.insert(v.end(), row.begin(), row.end());
  return v;
}

int main() {
  srand(0);

  // Example tiny 5x5 "image -> its just an random image don'nt worry about it "
  vector<vector<double>> img = {{1, 0, 1, 0, 1},
                                {0, 1, 0, 1, 0},
                                {1, 0, 1, 0, 1},
                                {0, 1, 0, 1, 0},
                                {1, 0, 1, 0, 1}};

  // 3x3 conv filter (random)
  vector<vector<double>> kernel = {
      {0.1, -0.2, 0.1}, {0.0, 0.2, -0.1}, {-0.1, 0.0, 0.1}};

  // Fully connected weights (flattened conv → 1 output)
  vector<double> fc_weights(9, 0.1); // random small weights
  double bias = 0.0, lr = 0.1;

  // Forward pass
  auto feature = conv2d(img, kernel);
  auto flat = flatten(feature);
  double sum = bias;
  for (size_t i = 0; i < flat.size(); i++)
    sum += flat[i] * fc_weights[i];
  double output = sigmoid(sum);

  // Assume label = 1
  double target = 1.0;
  double error = target - output;

  // Backprop to FC layer
  double delta = error * dsigmoid(output);
  for (size_t i = 0; i < fc_weights.size(); i++)
    fc_weights[i] += lr * delta * flat[i];
  bias += lr * delta;

  // Backprop to kernel (simple case)
  int idx = 0;
  for (int i = 0; i < kernel.size(); i++)
    for (int j = 0; j < kernel[0].size(); j++)
      kernel[i][j] += lr * delta * feature[i][j]; // simplified

  cout << "Output: " << output << "\n";
  cout << "Updated bias: " << bias << "\n";
}
