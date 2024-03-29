"""
Author: Baichuan Zhou, Lei Huang
Institution: Beihang University
Date: Spring, 2024
Course: Undergraduate-level Deep Learning
Website: https://baichuanzhou.github.io/, https://huangleibuaa.github.io/
"""

from abc import ABC, abstractmethod
import numpy as np

from neural_networks.layers import initialize_loss
from neural_networks.optimizers import initialize_optimizer
from neural_networks.layers import FullyConnected
from collections import OrderedDict
import pickle
from tqdm import tqdm

# imports for typing only
from neural_networks.utils.data_structures import AttrDict
from neural_networks.datasets import Dataset
from typing import Any, Dict, List, Sequence, Tuple


def initialize_model(name, loss, layer_args, optimizer_args, logger=None, seed=None):
    return NeuralNetwork(
        loss=loss,
        layer_args=layer_args,
        optimizer_args=optimizer_args,
        logger=logger,
        seed=seed,
    )


class NeuralNetwork(ABC):
    def __init__(
            self,
            loss: str,
            layer_args: Sequence[AttrDict],
            optimizer_args: AttrDict,
            logger=None,
            seed: int = None,
    ) -> None:

        self.n_layers = len(layer_args)
        self.layer_args = layer_args
        self.logger = logger
        self.epoch_log = {"loss": {}, "error": {}}

        self.loss = initialize_loss(loss)
        self.optimizer = initialize_optimizer(**optimizer_args)
        self._initialize_layers(layer_args)

    def _initialize_layers(self, layer_args: Sequence[AttrDict]) -> None:
        self.layers = []
        for l_arg in layer_args[:-1]:
            l = FullyConnected(**l_arg)
            self.layers.append(l)

    def _log(self, loss: float, error: float, validation: bool = False) -> None:

        if self.logger is not None:
            if validation:

                self.epoch_log["loss"]["validate"] = round(loss, 4)
                self.epoch_log["error"]["validate"] = round(error, 4)
                self.logger.push(self.epoch_log)
                self.epoch_log = {"loss": {}, "error": {}}
            else:
                self.epoch_log["loss"]["train"] = round(loss, 4)
                self.epoch_log["error"]["train"] = round(error, 4)

    def save_parameters(self, epoch: int) -> None:
        parameters = {}
        for i, l in enumerate(self.layers):
            parameters[i] = l.parameters
        if self.logger is None:
            raise ValueError("Must have a logger")
        else:
            with open(
                    self.logger.save_dir + "parameters_epoch{}".format(epoch), "wb"
            ) as f:
                pickle.dump(parameters, f)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播，传播通过神经网络的所有层

        Parameters
        ----------
        X  输入数据张量，X.shape[1]必须与权重的shape相匹配

        Returns
        -------
        每一层输出的shape必须与后一层相匹配，最后输出(batch_size, num_classes)，所以不要计算Loss
        """
        ### YOUR CODE HERE ###
        # 前向传播，提示，应该只需要一个for循环即可，两行
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, dLoss: np.ndarray) -> None:
        """反向传播
        这个函数应该由一个简单的for循环即可实现（大部分工作已经在之前的代码中实现）

        Parameters
        ----------
        dLoss   损失Loss对模型最终输出的梯度
        """
        ### YOUR CODE HERE ###
        # 将梯度反传，提示，应该只需要一个for循环即可，两行
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)
        return dLoss


    def update(self) -> None:
        """One step of gradient update using the derivatives calculated by
        `self.backward`.

        Parameters
        ----------
        epoch  the epoch we are currently on
        """
        param_log = {}
        for i, layer in enumerate(self.layers):
            for param_name, param in layer.parameters.items():
                if param_name != "null":  # FIXME: possible change needed to `is not`
                    param_grad = layer.gradients[param_name]
                    # Optimizer needs to keep track of layers
                    delta = self.optimizer.update(
                        param_name + str(i), param, param_grad
                    )
                    layer.parameters[param_name] -= delta
                    if self.logger is not None:
                        param_log["{}{}".format(param_name, i)] = {}
                        param_log["{}{}".format(param_name, i)]["max"] = np.max(param)
                        param_log["{}{}".format(param_name, i)]["min"] = np.min(param)
            layer.clear_gradients()
        self.epoch_log["params"] = param_log

    def error(self, target: np.ndarray, out: np.ndarray) -> float:
        """Only calculate the error of the model's predictions given `target`.

        For classification tasks,
            error = 1 - accuracy

        For regression tasks,
            error = mean squared error

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on features corresponding to
                `target`

        Returns
        -------
        the error of the model given the training inputs and targets
        """
        # classification error
        if self.loss.name == "cross_entropy":
            predictions = np.argmax(out, axis=1)
            target_idxs = np.argmax(target, axis=1)
            error = np.mean(predictions != target_idxs)

        # Error!
        else:
            raise NotImplementedError(
                "Error for {} loss is not implemented".format(self.loss)
            )

        return error

    def train(self, dataset: Dataset, epochs: int) -> None:

        # Initialize output layer
        args = self.layer_args[-1]
        args["n_out"] = dataset.out_dim
        output_layer = FullyConnected(**args)
        self.layers.append(output_layer)

        for i in range(epochs):
            training_loss = []
            training_error = []
            for _ in tqdm(range(dataset.train.samples_per_epoch)):
                X, Y = dataset.train.sample()
                logits = self.forward(X)
                L = self.loss(logits, Y)
                dL = self.loss.backward()
                self.backward(dL)
                error = self.error(Y, logits)
                self.update()
                training_loss.append(L)
                training_error.append(error)
            training_loss = np.mean(training_loss)
            training_error = np.mean(training_error)
            self._log(training_loss, training_error)

            validation_loss = []
            validation_error = []
            for _ in range(dataset.validate.samples_per_epoch):
                X, Y = dataset.validate.sample()
                logits = self.forward(X)
                L = self.loss.forward(logits, Y)
                error = self.error(Y, logits)
                validation_loss.append(L)
                validation_error.append(error)
            validation_loss = np.mean(validation_loss)
            validation_error = np.mean(validation_error)
            self._log(validation_loss, validation_error, validation=True)

            # print("Example target: {}".format(Y[0]))
            # print("Example prediction: {}".format([round(x, 4) for x in Y_hat[0]]))
            print(
                "Epoch {} Training Loss: {} Training Accuracy: {} Val Loss: {} Val Accuracy: {}".format(
                    i,
                    round(training_loss, 4),
                    round(1 - training_error, 4),
                    round(validation_loss, 4),
                    round(1 - validation_error, 4),
                )
            )

    def test(
            self, dataset: Dataset, save_predictions: bool = False
    ) -> Dict[str, List[np.ndarray]]:

        test_log = {"loss": [], "error": []}
        if save_predictions:
            test_log["prediction"] = []
        for _ in range(dataset.test.samples_per_epoch):
            X, Y = dataset.test.sample()
            Y_hat, L = self.predict(X, Y)
            error = self.error(Y, Y_hat)
            test_log["loss"].append(L)
            test_log["error"].append(error)
            if save_predictions:
                test_log["prediction"] += [x for x in Y_hat]
        test_loss = np.mean(test_log["loss"])
        test_error = np.mean(test_log["error"])
        print(
            "Test Loss: {} Test Accuracy: {}".format(
                round(test_loss, 4), round(1 - test_error, 4)
            )
        )
        if save_predictions:
            with open(self.logger.save_dir + "test_predictions.p", "wb") as f:
                pickle.dump(test_log, f)
        return test_log

    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a forward and backward pass to calculate the predictions and
        loss of the neural network on the given data.

        Parameters
        ----------
        X  input features
        Y  targets (same length as `X`)

        Returns
        -------
        a tuple of the prediction and loss
        """
        ### YOUR CODE HERE ###
        # Do a forward pass. Maybe use a function you already wrote?
        prediction = self.forward(X)
        # Get the loss. Remember that the `backward` function returns the loss.
        loss = self.loss(prediction, Y)
        return prediction, loss
