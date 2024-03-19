"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""

from abc import ABC, abstractmethod


class Module(ABC):
    """Abstract class defining the common interface for all modules."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass

    @abstractmethod
    def backward(self, dY):
        pass
