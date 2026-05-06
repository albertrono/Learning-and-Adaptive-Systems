import numpy as np

# Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# MLP Class 
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr

        # Xavier-style weight initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Layer 1: Input → Hidden
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        # Layer 2: Hidden → Output
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients (backpropagated error)
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent weight update
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f'Epoch {epoch:4d}  |  Loss: {loss:.6f}')

# XOR Example
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],   [1],   [1],   [0]])

mlp = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.5)
mlp.train(X, y, epochs=5000)
print('Predictions:', mlp.forward(X).round())
