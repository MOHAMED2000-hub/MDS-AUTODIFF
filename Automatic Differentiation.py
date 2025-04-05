import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to compute derivative using PyTorch
def compute_derivative(f, x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = f(x_tensor)
    y.backward(torch.ones_like(x_tensor))
    return x_tensor.detach().numpy(), x_tensor.grad.detach().numpy()

# Function to compute second derivative
def compute_second_derivative(f, x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = f(x_tensor)
    grad_1 = torch.autograd.grad(y, x_tensor, grad_outputs=torch.ones_like(x_tensor), create_graph=True)[0]
    grad_2 = torch.autograd.grad(grad_1, x_tensor, grad_outputs=torch.ones_like(x_tensor))[0]
    return x_tensor.detach().numpy(), grad_2.detach().numpy()

# Define functions
f_sin = torch.sin
f_linear = lambda x: 3*x + 5  # Linear function 3x + 5

# Define x ranges
x_sin = np.linspace(0, 2 * np.pi, 20)
x_linear = np.linspace(0, 10, 20)

# Compute derivatives
x_sin_vals, sin_derivative = compute_derivative(f_sin, x_sin)
_, sin_second_derivative = compute_second_derivative(f_sin, x_sin)
x_linear_vals, linear_derivative = compute_derivative(f_linear, x_linear)

# Plot results
plt.figure(figsize=(12, 5))

# Plot sin(x) derivative
plt.subplot(1, 3, 1)
plt.plot(x_sin_vals, sin_derivative, label="sin'(x)", color="b")
plt.title("First Derivative of sin(x)")
plt.xlabel("x")
plt.ylabel("Derivative")
plt.legend()

# Plot linear function derivative
plt.subplot(1, 3, 2)
plt.plot(x_linear_vals, linear_derivative, label="Linear function derivative", color="g")
plt.title("Derivative of Linear Function")
plt.xlabel("x")
plt.ylabel("Derivative")
plt.legend()

# Plot second derivative of sin(x)
plt.subplot(1, 3, 3)
plt.plot(x_sin_vals, sin_second_derivative, label="sin''(x)", color="r")
plt.title("Second Derivative of sin(x)")
plt.xlabel("x")
plt.ylabel("Second Derivative")
plt.legend()

plt.tight_layout()
plt.show()
