import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, n_input, n_output, conv1_size, conv2_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        conv1_size, int - number of filters in the 1st conv layer
        conv2_size, int - number of filters in the 2nd conv layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [
            ConvolutionalLayer(3, conv1_size, filter_size=3, padding=0),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_size, conv2_size, filter_size=3, padding=0),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(n_input=conv2_size, n_output=n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        loss, grad = softmax_with_cross_entropy(pred, y)

        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
            if type(layer) == FullyConnectedLayer and self.reg:
                for param in layer.params().values():
                    l2_loss, l2_grad = l2_regularization(param.value, self.reg)
                    param.grad += l2_grad
                    loss += l2_loss
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        return np.argmax(softmax(pred), axis=1)

    def params(self):
        return {
            '{}_{}'.format(param_name, idx): param_value
            for idx, layer in enumerate(self.layers)
            for param_name, param_value in layer.params().items()
        }
