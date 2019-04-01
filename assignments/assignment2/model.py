import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden_layers = [
            FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        pred = X.copy()
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        for layer in self.hidden_layers:
            pred = layer.forward(pred)

        loss, grad = softmax_with_cross_entropy(pred, y)

        for layer in self.hidden_layers[::-1]:
            grad = layer.backward(grad)

        if self.reg:
            for param in self.params().values():
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
        for layer in self.hidden_layers:
            pred = layer.forward(pred)
        return np.argmax(softmax(pred), axis=1)

    def params(self):
        return {
            '{}_{}'.format(param_name, idx): param_value
            for idx, layer in enumerate(self.hidden_layers)
            for param_name, param_value in layer.params().items()
        }
