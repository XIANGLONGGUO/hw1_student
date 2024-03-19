"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""
from neural_networks.layers.module import Module
from neural_networks.layers.activations import initialize_activation
from neural_networks.layers.initializations import initialize_weights

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Callable


class FullyConnected(Module):
    """这个模块实现最基本的Affine运算+激活函数，Y = XW + b，其中X是一个二维张量，W为权重（同为二维张量），b为bias
    """

    def __init__(
            self,
            n_out: int,
            activation: str = "relu",
            weight_init: str = "xavier_uniform"
    ):
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        self.init_weights = initialize_weights(weight_init, activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """初始化所有参数。（weights，biases）"""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###
        W = ...  # 利用init_weights初始化权重
        b = ...  # 初始化偏置

        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = ...   # 反向传播你需要什么？
        # 把梯度初始化为0即可
        self.gradients: OrderedDict = ...

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播：输入张量X是一个二维张量，与参数W进行矩阵乘法后，加上bias后，通过激活函数，得到输出。
        把所有反向传播中可能用到的张量或参数存入cache中。

        Parameters
        ----------
        X  输入张量的shape (batch_size, input_dim)

        Returns
        -------
        输出张量 (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        # 计算矩阵相乘，与经过激活函数后的值


        # 把反向传播需要的东西存储在cache中


        ### END YOUR CODE ###

        return ...

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """反向传播实现。
        需要计算损失Loss对三个参数的梯度
            1. 权重 W
            2. bias b
            3. 本层的输入张量Z

        Parameters
        ----------
        dLdY  损失Loss对张量本层输出值的梯度（和输出张量的shape相同，(batch_size, output_dim)）

        Returns
        -------
        损失Loss对张量本层输出值的梯度dLdX和输入张量的shape相同，(batch_size, input_dim)）
        """
        ### BEGIN YOUR CODE ###

        # 从cache中获取需要的张量

        # 计算梯度


        # 把梯度存在self.gradients中
        # 权重W的梯度应当存在self.gradients['W']中，偏置b同理

        ### END YOUR CODE ###

        return ...

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
            self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]
