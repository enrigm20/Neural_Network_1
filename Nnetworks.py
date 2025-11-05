import numpy as np


def sigmoid(x):                         #activation function based on weights
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):           #linearizes activation function (neuron sensitivity)
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.rand(2, 2)  # 2 inputs and 2 hidden neurons
        self.W2 = np.random.rand(2, 1)  # 2 inputs and 1 output
        self.learn_rate = 0.6                   

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.W1)) #multiplies weighs matrix(x)  * possible ROS combinations  (2 weighs for 2 neurons)
        self.output = sigmoid(np.dot(self.hidden, self.W2))   #used to combine the 2 hidden neuron values into 1 output (2 weighs for 1 output)
        return self.output

    def train(self, X, y, epochs=10000):
        for i in range(epochs):
            self.forward(X)
            error = y - self.output   #expected - output (overall error)
            d_output = error * sigmoid_derivative(self.output)  #gives adujsted error, based on neuron sensitivity (how much the neurons need to adjust)
            d_hidden = d_output.dot(self.W2.T) * sigmoid_derivative(self.hidden) # backpropagates the output error to  hidden layer and scales with hidden neurons sensitivity
            self.W2 += self.hidden.T.dot(d_output) * self.learn_rate  #adjusts output weight, based on the adjsted error and learning rate
            self.W1 += X.T.dot(d_hidden) * self.learn_rate  #updated weights are adjusted

    def predict(self, X):
        return self.forward(X)

X = np.array([[0,0],[0,1],[1,0],[1,1]])  #possible ROS combinations inputs (XOR)
y = np.array([[0],[1],[1],[0]])          #possible ROS combinations outputs (XOR)

nn = NeuralNetwork()   
nn.train(X, y, epochs=10000)   #x and y correspong to XOR values outputs

print("\nFinal residues:")   #printing results
for i, sample in enumerate(X):
    pred = nn.predict(sample.reshape(1, -1))
    print(f"Input: {sample} | Pred: {pred[0][0]:.3f} | Real: {y[i][0]}")
