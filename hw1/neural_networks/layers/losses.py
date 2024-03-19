"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""
from collections import OrderedDict

import numpy as np
from abc import ABC, abstractmethod
from neural_networks.layers import SoftMax


class Loss(ABC):
    def __call__(self, Y, Y_hat):
        return self.forward(Y, Y_hat)

    @abstractmethod
    def forward(self, Y, Y_hat):
        pass

    @abstractmethod
    def backward(self):
        pass


def initialize_loss(name: str) -> Loss:
    if name == "cross_entropy":
        return CrossEntropyLoss()
    else:
        raise NotImplementedError("{} loss is not implemented".format(name))


class CrossEntropyLoss(Loss):
    """Cross entropy loss function."""
    name = "cross_entropy"

    def __init__(self) -> None:
        self.cache = None
        self.softmax = SoftMax()

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> float:
        return self.forward(logits, labels)

    def forward(self, logits: np.ndarray, Y) -> float:
        """利用CrossEntropy计算logits和真实标签Y(one-hot编码)的损失，logits是模型的最终输出，
        应当经过SoftMax激活函数后与Y，利用CrossEntropy计算损失

        Parameters
        ----------
        logits  模型的最终输出 (batch_size, num_classes)
        Y       标签的one-hot编码 (batch_size, num_classes)

        Returns
        -------
        Loss，一个浮点数
        """
        if self.cache is None:
            self.cache = OrderedDict({'Y': np.zeros_like(Y), 'Y_hat': np.zeros_like(Y)})
        ### YOUR CODE HERE ###
        Y_hat = self.softmax(logits)
        self.cache['Y'] = Y
        self.cache['Y_hat'] = Y_hat
        return np.mean(-np.log(Y_hat[Y == 1]))

    def backward(self) -> np.ndarray:
        """CrossEntropyLoss的反向传播实现

        Returns
        -------
        损失Loss对logits的梯度 (batch_size, num_classes)
        """
        ### YOUR CODE HERE ###
        Y, Y_hat = self.cache['Y'], self.cache['Y_hat']
        dY = np.zeros_like(Y)
        dY[Y == 1] = -1 / Y_hat[Y == 1] / Y.shape[0]
        dlogits = self.softmax.backward(dY)
        return dlogits
