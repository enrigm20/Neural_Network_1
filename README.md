# XOR Neural Network from Scratch using NumPy

This project implements a simple feedforward neural network from scratch using only NumPy.  
The network is trained to solve the XOR problem using forward propagation and backpropagation.

---

# XOR Truth Table

| Input 1 | Input 2 | Output |
|----------|----------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The XOR problem is a classic example because it is not linearly separable, so a hidden layer is required.

---

# Network Architecture

- 2 input neurons
- 2 hidden neurons
- 1 output neuron

```text
Input Layer (2)
      ↓
Hidden Layer (2)
      ↓
Output Layer (1)
```

---

# Activation Function

The network uses the sigmoid activation function:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

Derivative:

```python
def sigmoid_derivative(x):
    return x * (1 - x)
```

---

# How It Works

## 1. Initialize Weights

Random weights are generated for:
- Input → Hidden layer
- Hidden → Output layer

```python
self.W1 = np.random.rand(2, 2)
self.W2 = np.random.rand(2, 1)
```

---

## 2. Forward Propagation

Hidden layer:

```python
self.hidden = sigmoid(np.dot(X, self.W1))
```

Output layer:

```python
self.output = sigmoid(np.dot(self.hidden, self.W2))
```

---

## 3. Error Calculation

```python
error = y - self.output
```

---

## 4. Backpropagation

Output layer adjustment:

```python
d_output = error * sigmoid_derivative(self.output)
```

Hidden layer adjustment:

```python
d_hidden = d_output.dot(self.W2.T) * sigmoid_derivative(self.hidden)
```

---

## 5. Weight Updates

```python
self.W2 += self.hidden.T.dot(d_output) * self.learn_rate
self.W1 += X.T.dot(d_hidden) * self.learn_rate
```

---

# Training

The network is trained for 10,000 epochs.

```python
nn.train(X, y, epochs=10000)
```

Learning rate:

```python
self.learn_rate = 0.6
```

---

# Example Output

```text
Final residues:
Input: [0 0] | Pred: 0.021 | Real: 0
Input: [0 1] | Pred: 0.972 | Real: 1
Input: [1 0] | Pred: 0.973 | Real: 1
Input: [1 1] | Pred: 0.028 | Real: 0
```

Values close to:
- 0 = FALSE
- 1 = TRUE

---

# Requirements

Install NumPy:

```bash
pip install numpy
```

---

# Run

```bash
python neural_network.py
```

---

# Educational Purpose

This project is intended for educational purposes to demonstrate:
- Feedforward neural networks
- Sigmoid activation
- Backpropagation
- Weight optimization using gradient descent
- Solving XOR with a hidden layer

```
