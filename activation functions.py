import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def step_function(x):
    return np.where(x >= 0, 1, 0)

def linear_function(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    return np.log(1 + np.exp(x))

def swish(x):
    return x * sigmoid(x)

def softsign(x):
    return x / (1 + np.abs(x))

# Derivatives of activation functions
def step_function_derivative(x):
    return np.zeros_like(x)  # Derivative is zero almost everywhere

def linear_function_derivative(x):
    return np.ones_like(x)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu_derivative(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def softsign_derivative(x):
    return 1 / (1 + np.abs(x)) ** 2

# Generate input values ranging from -10 to 10 with small increments
x = np.linspace(-10, 10, 1000)

# Plot activation functions and their derivatives
activation_functions = [
    (step_function, step_function_derivative, "Step Function"),
    (linear_function, linear_function_derivative, "Linear Function"),
    (sigmoid, sigmoid_derivative, "Sigmoid"),
    (tanh, tanh_derivative, "Tanh"),
    (relu, relu_derivative, "ReLU"),
    (leaky_relu, leaky_relu_derivative, "Leaky ReLU"),
    (elu, elu_derivative, "ELU"),
    (softplus, softplus_derivative, "Softplus"),
    (swish, swish_derivative, "Swish"),
    (softsign, softsign_derivative, "Softsign")
]

plt.figure(figsize=(15, 25))

for i, (func, derivative_func, title) in enumerate(activation_functions):
    y = func(x)
    dy_dx = derivative_func(x)
    
    # Print outputs for a sample of values
    print(f"\n{title} Outputs:")
    print(y[:5], "...")  # Displaying the first 5 values
    print(f"{title} Derivatives:")
    print(dy_dx[:5], "...")  # Displaying the first 5 derivative values
    
    # Plot activation function output
    plt.subplot(len(activation_functions), 2, 2*i+1)
    plt.plot(x, y, label=title, color='blue')
    plt.title(f"{title} Output")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

    # Plot derivative
    plt.subplot(len(activation_functions), 2, 2*i+2)
    plt.plot(x, dy_dx, label=f"{title} Derivative", color='red')
    plt.title(f"{title} Derivative")
    plt.xlabel("Input")
    plt.ylabel("Derivative")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
