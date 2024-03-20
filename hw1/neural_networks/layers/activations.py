"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""
from neural_networks.layers.module import Module

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict


class Activation(Module, ABC):
    """Abstract class defining the common interface for all activation methods."""
    name: str = "activation"


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "identity":
        return Identity()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Identity(Activation):
    name = "identity"
    """激活函数：f(z) = z"""

    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        Z=np.maximum(0,Z)
        return Z

    def backward(self, dY: np.ndarray) -> np.ndarray:
        dY=np.where(dY > 0, dY, 0)
        return dY


class Sigmoid(Activation):
    name = "sigmoid"
    """激活函数：f(z) = 1 / (1 + exp(-z))"""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        sigmoid = 1 / (1 + np.exp(-Z))
        self.cache = sigmoid
        return sigmoid

    def backward(self, dY: np.ndarray) -> np.ndarray:
        sigmoid = self.cache
        return dY * sigmoid * (1 - sigmoid)


class TanH(Activation):
    name = "tanh"
    """激活函数：f(z) = (exp(z) - exp(-z) / (exp(z) + exp(-z))"""
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Tanh激活函数的前向传播实现:
        f(z) = (exp(z) - exp(-z) / (exp(z) + exp(-z))
        把所有反向传播中可能用到的张量或参数存入cache中
        Parameters
        ----------
        Z  输入张量Z

        Returns
        -------
        f(z)
        """
        if self.cache is None:
            self.cache = OrderedDict({'Z': np.zeros_like(Z)})
        self.cache['Z'] = Z
        return 2 / (1 + np.exp(-2 * Z)) - 1

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Tanh激活函数的反向传播实现.

        Parameters
        ----------
        dY  损失Loss对张量激活值f(Z)的梯度（和输入张量Z的shape相同）

        Returns
        -------
        损失Loss对张量Z的梯度
        ** 不要使用for循环实现 **
        """
        Z = self.cache['Z']
        fn = self.forward(Z)
        return dY * (1 - fn ** 2)

import numpy as np
from collections import OrderedDict

class ReLU(Activation):
    name = "relu"
    """f(x) = x if x > 0 else 0"""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        if self.cache is None:
            self.cache = OrderedDict({'Z': np.zeros_like(Z)})
        self.cache['Z'] = Z
        return np.maximum(0, Z)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        Z = self.cache['Z']
        dZ = np.where(Z > 0, dY, 0)
        return dZ


class SoftMax(Activation):
    name = "softmax"

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:

        if self.cache is None:
            self.cache = OrderedDict({'Z': np.zeros_like(Z)})
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        softmax = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
        self.cache['Z'] = softmax
        return softmax

    def backward(self, dY: np.ndarray) -> np.ndarray:
        softmax = self.cache['Z']
        return softmax * (dY - np.sum(softmax * dY, axis=-1, keepdims=True))
