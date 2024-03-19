"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""

import numpy as np
from abc import ABC, abstractmethod


def initialize_optimizer(
    name,
    lr,
    momentum=None,
    clip_norm=None
):
    if name == "SGD":
        return SGD(
            lr=lr,
            momentum=momentum,
            clip_norm=clip_norm
        )
    else:
        raise NotImplementedError


class SGD:
    def __init__(
        self,
        lr,
        momentum=0.0,
        clip_norm=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        delta = (
            self.momentum * self.cache[param_name]
            + self.lr * param_grad
        )
        self.cache[param_name] = delta
        return delta
