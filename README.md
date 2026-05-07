# ROS Agent Detection Neural Network using NumPy

This project implements a simple feedforward neural network from scratch using only NumPy.  
The neural network is designed as a conceptual model for ROS agent detection using binary signal combinations.

The project demonstrates:
- Feedforward neural networks
- Hidden layers
- Sigmoid activation functions
- Backpropagation
- Weight optimization using gradient descent

---

# Overview

Reactive Oxygen Species (ROS) are chemically reactive molecules involved in oxidative stress and cellular signaling.  

In this simplified model:
- Inputs represent possible ROS-related signal combinations
- The neural network predicts whether the signal pattern corresponds to a normal or abnormal ROS-related response

This is a conceptual educational example inspired by biological signaling systems.

---

# Dataset

| Signal 1 | Signal 2 | Output |
|----------|----------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The dataset follows an XOR-like structure to demonstrate nonlinear classification.

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

```python id="4wc0mf"
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

Derivative:

```python id="3h40w8"
def sigmoid_derivative(x):
    return x * (1 - x)
```

---

# How It Works

## 1. Initialize Weights

Random weights are generated for:
- Input → Hidden layer
- Hidden → Output layer

```python id="pv5o9u"
self.W1 = np.random.rand(2, 2)
self.W2 = np.random.rand(2, 1)
```

---

## 2. Forward Propagation

Hidden layer:

```python id="10qlyd"
self.hidden = sigmoid(np.dot(X, self.W1))
```

Output layer:

```python id="fd5pfc"
self.output = sigmoid(np.dot(self.hidden, self.W2))
```

---

## 3. Error Calculation

```python id="j5alr5"
error = y - self.output
```

---

## 4. Backpropagation

Output layer adjustment:

```python id="3xfw7d"
d_output = error * sigmoid_derivative(self.output)
```

Hidden layer adjustment:

```python id="qk04qo"
d_hidden = d_output.dot(self.W2.T) * sigmoid_derivative(self.hidden)
```

---

## 5. Weight Updates

```python id="3m96t6"
self.W2 += self.hidden.T.dot(d_output) * self.learn_rate
self.W1 += X.T.dot(d_hidden) * self.learn_rate
```

---

# Training

The network is trained for 10,000 epochs.

```python id="bp64lc"
nn.train(X, y, epochs=10000)
```

Learning rate:

```python id="ttjlwm"
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
- 0 = Normal response
- 1 = Abnormal ROS-related response

---

# Requirements

Install NumPy:

```bash id="t5l5l9"
pip install numpy
```

---

# Run

```bash id="y3dr1f"
python neural_network.py
```

---

# Educational Purpose

This project is intended for educational purposes to demonstrate:
- Feedforward neural networks
- Nonlinear classification
- Backpropagation
- Gradient descent optimization
- Conceptual ROS signal detection systems

```
