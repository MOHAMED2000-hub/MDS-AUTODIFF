import matplotlib.pyplot as plt
import torch
from torch import nn


class LinearFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1,
                                out_features=1)

    def forward(self, x):
        return self.linear(x)


def differentiate_sin_1():
    # Define independent variable
    x = torch.linspace(start=0.0, end=2 * torch.pi, steps=20,
                       requires_grad=True)
    # Execute forward evaluation
    y = torch.sin(x)
    # Plot the function
    plt.plot(x.detach(), y.detach())

    # Take sum for a single value result
    d_sum = y.sum()
    # Calculate gradients
    d_sum.backward()
    # Plot the derivative
    plt.plot(x.detach(), x.grad.detach())

    # Show results (needed in PyCharm)
    plt.show()


def differentiate_linear():
    # Define independent variable
    x = torch.linspace(start=0.0, end=10.0, steps=100).reshape(100, 1)
    x.requires_grad = True

    # Instantiate parametric linear function
    func = LinearFunc()
    # Execute forward evaluation
    y = func(x)
    # Plot the function
    plt.plot(x.detach(), y.detach())

    # Take sum for a single value result
    d_sum = y.sum()
    # Calculate gradients
    d_sum.backward()
    # Plot the derivative
    plt.plot(x.detach(), x.grad.detach())

    # Show results (needed in PyCharm)
    plt.show()


def differentiate_sin_2():
    # Define independent variable
    x = torch.linspace(start=0.0, end=2 * torch.pi, steps=20,
                       requires_grad=True)
    # Execute forward evaluation
    y = torch.sin(x)
    # Plot the function
    plt.plot(x.detach(), y.detach())

    # Take sum for a single value result
    d_sum = y.sum()
    # Calculate gradients with autograd.grad
    # Create a retain graph for further calculations
    dy = torch.autograd.grad(outputs=d_sum, inputs=x,
                             create_graph=True,
                             retain_graph=True)[0]
    # Plot the derivative
    plt.plot(x.detach(), dy.detach())

    # Take sum for a single value result
    ddy_sum = dy.sum()
    # Calculate the second derivative with autograd.grad
    ddy = torch.autograd.grad(outputs=ddy_sum, inputs=x)[0]
    # Plot the second derivative
    plt.plot(x.detach(), ddy.detach())

    # Show results (needed in PyCharm)
    plt.show()


def main():
    exercise_number = 3

    if exercise_number == 1:
        # 1. Differentiation of sin(x)
        differentiate_sin_1()
    elif exercise_number == 2:
        # 2. Differentiation for parametric function
        differentiate_linear()
    else:
        # 3. Create backward graph to calculate second derivative
        differentiate_sin_2()


if __name__ == "__main__":
    main()
