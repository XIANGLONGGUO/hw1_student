# Activation Function Implementations:

Implementation of `layers.activations.Identity`:

```python
class Identity(Activation):
    name = "identity"
    """激活函数：f(z) = z"""

    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z

    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY

```

Implementation of `layers.activations.Sigmoid`:

```python
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

```

Implementation of `layers.activations.ReLU`:

```python
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

```

Implementation of `layers.activations.SoftMax`:

```python
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

```


# Layer Implementations:

Implementation of `layers.fully_connected.FullyConnected`:

```python
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
        W = self.init_weights((self.n_in, self.n_out))  # 利用init_weights初始化权重
        b = b = np.zeros((1, self.n_out))  # 初始化偏置

        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict =OrderedDict({"Z": None, "X": None})  # 反向传播你需要什么？
        # 把梯度初始化为0即可
        self.gradients: OrderedDict =  OrderedDict({'W':dW,'b':b}) 
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
        # 计算矩阵相乘，与经过激活函数后的值
        if self.n_in is None:
            self._init_parameters(X.shape)
        z=np.dot(X,self.parameters['W'])+self.parameters['b']
        out=self.activation(z)
        # 把反向传播需要的东西存储在cache中
        self.cache['Z']=z

        self.cache['X']=X
        return out

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
        z,X=self.cache['Z'],self.cache['X']
        # 计算梯度
        z,X=self.cache['Z'],self.cache['X']
        dLdZ=self.activation.backward(dLdY)
        dX=dLdZ.dot(self.parameters['W'].T)

        # 把梯度存在self.gradients中
        # 权重W的梯度应当存在self.gradients['W']中，偏置b同理
        self.gradients['W']=X.T.dot(dLdZ)
        self.gradients['b']=np.sum(dLdZ,axis=0)
        ### END YOUR CODE ###


        return dX
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

```


# Loss Function Implementations:

Implementation of `layers.losses.CrossEntropyLoss`:

```python
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

```


# Model Implementations:

Implementation of `models.NeuralNetwork.forward`:

```python
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

```

Implementation of `models.NeuralNetwork.backward`:

```python
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

```

Implementation of `models.NeuralNetwork.predict`:

```python
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

```

