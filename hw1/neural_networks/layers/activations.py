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
        """f(z) = z 的前向传播实现

        Parameters
        ----------
        Z  传入激活函数前的张量

        Returns
        -------
        经过激活函数后的输出值，在Identity函数的情况下，返回传入前的张量Z
        """
        return Z

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """f(z) = z 的反向传播实现

        Parameters
        ----------
        dY  损失Loss对张量激活值f(Z)的梯度（和输入张量Z的shape相同）

        Returns
        -------
        损失Loss对张量Z的梯度
        """
        return dY


class Sigmoid(Activation):
    name = "sigmoid"
    """激活函数：f(z) = 1 / (1 + exp(-z))"""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数的前向传播实现:
        f(z) = 1 / (1 + exp(-z))
        把所有反向传播中可能用到的张量或参数存入cache中

        Parameters
        ----------
        Z  输入张量Z

        Returns
        -------
        f(z) = 1 / (1 + exp(-z))
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数的反向传播实现

        Parameters
        ----------
        dY  损失Loss对张量激活值f(Z)的梯度（和输入张量Z的shape相同）

        Returns
        -------
        损失Loss对张量Z的梯度
        ** 不要使用for循环实现 **
        """
        ### YOUR CODE HERE ###
        return ...


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


class ReLU(Activation):
    name = "relu"
    """f(x) = x if x > 0 else 0"""
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """ReLU激活函数的前向传播实现
        f(z) = z if z > 0 else 0
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
        ### YOUR CODE HERE ###
        return ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """ReLU激活函数的反向传播实现

        Parameters
        ----------
        dY  损失Loss对张量激活值f(Z)的梯度（和输入张量Z的shape相同）

        Returns
        -------
        损失Loss对张量Z的梯度
        ** 不要使用for循环实现 **
        """
        ### YOUR CODE HERE ###

        # dY should be of shape: n x d
        # Elements in Z above zero have gradient 1, otherwise zero.
        return ...


class SoftMax(Activation):
    name = "softmax"

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """SoftMax激活函数的前向传播实现。（这个函数将与CrossEntropyLoss联合使用）
        Hint: 原始的方法有可能数值不稳定
        把所有反向传播中可能用到的张量或参数存入cache中
        Parameters
        ----------
        Z  输入张量Z

        Returns
        -------
        softmax(z)
        """
        if self.cache is None:
            self.cache = OrderedDict({'Z': np.zeros_like(Z)})
        ### YOUR CODE HERE ###
        return ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """SoftMax激活函数的反向传播实现

        Parameters
        ----------
        dY  损失Loss对张量激活值f(Z)的梯度（和输入张量Z的shape相同）

        Returns
        -------
        损失Loss对张量Z的梯度
        ** 不要使用for循环实现 **
        """
        ### YOUR CODE HERE ###
        return ...
