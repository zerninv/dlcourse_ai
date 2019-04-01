import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    return np.mean(-np.log(probs[range(probs.shape[0]), target_index]))


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    return reg_strength * np.sum(W * W), 2 * reg_strength * W


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    probs[range(probs.shape[0]), target_index] -= 1
    return loss, probs / probs.shape[0]


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X_mask = None

    def forward(self, X):
        self.X_mask = (X > 0)
        return X * self.X_mask

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out * self.X_mask

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.flatted_W = None
        self.X = None
        self.conv_X = None

    def forward(self, X):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        self.X = X.copy()

        padded_X = np.pad(
            self.X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant',
            constant_values=0
        )

        batch_size, height, width, channels = padded_X.shape
        self.flatted_W = np.reshape(self.W.value, (self.in_channels * (self.filter_size ** 2), self.out_channels))

        shift = self.filter_size
        out_height = height - shift + 1
        out_width = width - shift + 1

        self.conv_X = np.zeros((batch_size, out_height, out_width, channels * (self.filter_size ** 2)))
        predictions = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for i in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    self.conv_X[i, y, x] = padded_X[i, y:y + shift, x:x + shift].flatten()
                    predictions[i, y, x] = np.dot(self.conv_X[i, y, x], self.flatted_W) + self.B.value
        return predictions

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros(shape=(batch_size, height + 2 * self.padding, width + 2 * self.padding, channels))

        for i in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    exact_d_out = np.atleast_2d(d_out[i][y][x])
                    padded_dX = np.reshape(
                        np.dot(exact_d_out, self.flatted_W.T),
                        (self.filter_size, self.filter_size, channels)
                    )
                    dX[i, y: y + self.filter_size, x: x + self.filter_size] += padded_dX
                    self.W.grad += np.reshape(
                        np.dot(np.atleast_2d(self.conv_X[i][y][x]).T, exact_d_out),
                        newshape=self.W.grad.shape
                    )
                    self.B.grad += d_out[i][y][x]
        return dX[:, self.padding:height + self.padding, self.padding: width + self.padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X_shape = None
        self.max_indices = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = self.X_shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros(shape=(batch_size, out_height, out_width, channels))
        self.max_indices = np.zeros(shape=(batch_size, out_height, out_width, channels, 2), dtype=np.int)

        for i in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    y_start = y * self.stride
                    x_start = x * self.stride
                    pool = X[i, y_start: y_start + self.pool_size, x_start: x_start + self.pool_size]
                    flatten_pool = np.reshape(pool, newshape=(self.pool_size * self.pool_size, channels))
                    out[i][y][x] = np.max(flatten_pool, axis=0)
                    max_indices = np.argmax(flatten_pool, axis=0)
                    self.max_indices[i][y][x] = np.apply_along_axis(
                        lambda index: np.array([y_start + (index // self.pool_size), x_start + (index % self.pool_size)]),
                        axis=0,
                        arr=max_indices
                    ).T
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = d_out.shape
        dX = np.zeros(shape=self.X_shape)
        for i in range(batch_size):
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        m_y, m_x = self.max_indices[i, y, x, c]
                        dX[i, m_y, m_x, c] = d_out[i][y][x][c]
        return dX

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = self.X_shape
        return np.reshape(X, newshape=(batch_size, height * width * channels))

    def backward(self, d_out):
        batch_size, height, width, channels = self.X_shape
        return np.reshape(d_out, newshape=(batch_size, height, width, channels))

    def params(self):
        # No params!
        return {}
