import json
import numpy as np

# Task 1: JSON Configuration Parser
def parse_json_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Derivatives of activation functions
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

# Task 2: MLP Implementation
class MLP:
    def __init__(self, config):
        self.layers = config['layers']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.activation_funcs = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'relu': (relu, relu_derivative),
            'tanh': (tanh, tanh_derivative)
        }
        
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i]['neurons'], self.layers[i + 1]['neurons'])
            bias = np.random.randn(1, self.layers[i + 1]['neurons'])
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward_propagation(self, X):
        activations = [X]
        Zs = []
        
        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:  # No activation for the output layer
                activation_func = self.activation_funcs[self.layers[i + 1]['activation']][0]
                A = activation_func(Z)
            else:
                A = Z  # No activation for the output layer
            Zs.append(Z)
            activations.append(A)
        
        return activations, Zs
    
    def backward_propagation(self, X, y, activations, Zs):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        
        # Compute the delta for the output layer
        deltas[-1] = activations[-1] - y
        
        # Backpropagate through the layers
        for i in reversed(range(len(deltas) - 1)):
            activation_func_derivative = self.activation_funcs[self.layers[i + 1]['activation']][1]
            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * activation_func_derivative(Zs[i])
        
        # Compute the gradients
        for i in range(len(gradients_w)):
            gradients_w[i] = np.dot(activations[i].T, deltas[i]) / m
            gradients_b[i] = np.sum(deltas[i], axis=0, keepdims=True) / m
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y):
        print(self.epochs)
        for epoch in range(self.epochs):
            activations, Zs = self.forward_propagation(X)
            gradients_w, gradients_b = self.backward_propagation(X, y, activations, Zs)
            self.update_parameters(gradients_w, gradients_b)
            if epoch % 100 == 0:
                loss = np.mean(np.square(activations[-1] - y))
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

# Example usage:
config = parse_json_config('MLP.json')

# Create and train the MLP
mlp = MLP(config)

# Generate random example data (X: input features, y: target labels)
np.random.seed(42)
X = np.random.randn(100, config['layers'][0]['neurons'])
y = np.random.randn(100, config['layers'][-1]['neurons'])

# Train the network
mlp.train(X, y)

# Make predictions
predictions = mlp.predict(X)
print('Predictions:', predictions)
