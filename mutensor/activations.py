"""
    activations: implements activation functions and their derivatives



"""

from typing import Callable

import numpy as np

from mutensor.tensor import Tensor
from mutensor.layers import Layer

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def sigmoid(x : Tensor) -> Tensor:
    return 1 / (1+np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x)*(1-sigmoid(x))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid,sigmoid_prime)
