'''
Functions and classes to train a feedforward neural network.
'''

import numpy as np
import random

class FFNN():
    '''
    Class representing a simple feedforward neural network.
    '''

    def __init__(self, sizes):
        '''
        Initialises FFNN class.

        Args:
            sizes (list): A list containing the number of neurons
                in each layer. The length of the list is the number
                of layers in the network.
        '''
        # Initialise a random number generator
        rng = np.random.default_rng()

        # Set number of layers and size of each layer
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Generate biases for all layers except input layer
        # Each neuron has its own bias
        self.biases = [rng.standard_normal((y, 1)) for y in sizes[1:]]

        # Generate a list of weight matrices
        # Each matrix represents the weights between two consecutive layers
        # The jk-th weight is from the k-th neuron to the j-th neuron
        self.weights = [
            rng.standard_normal((y, x)) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def sigmoid(self, z: float) -> float:
        '''
        Applies the sigmoid function to an input z.

        args:
            z (float): A real-valued number.
        '''
        result = 1 / (1 + np.exp(-z))

        return result
    
    def sigmoid_prime(self, z: float) -> float:
        '''
        Returns the derivative of the sigmoid function
        for a given input z.

        args:
            z (float): A real-valued number.
        '''
        derivative = self.sigmoid(z) * (1 - self.sigmoid(z))

        return derivative

    def feedforward(self, a):
        '''
        Returns output of the network for the given input.
        
        args:
            a (ndarray): An array of size (n, 1).
        '''
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)

        return a
    
    def sgd(self,
            training_data,
            n_epochs: int,
            batch_size: int,
            learning_rate: float,
            test_data = None
    ) -> None:
        '''
        Trains a feedforward neural network using
        stochastic gradient descent.

        args:
            training_data (list): A list of tuples (x, y) where
                x is a (784, 1) array representing a training
                input, and y is a (10, 1) array representing a
                training label.
            n_epochs (int): Number of epochs to train for.
            batch_size (int): Number of training inputs in each mini-batch.
            learning_rate (float): The learning rate, a real-valued
                positive number.
            test_data (list): A list of tuples (x, y) representing
                the test data. If provided, the network will be evaluated
                on the test set after each epoch, and the result printed.
        '''
        n = len(training_data)
        if test_data:
            n_test = len(test_data)
        for j in range(n_epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)
            ]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print(
                    f'Epoch {j}: {self.evaluate(test_data)} / {n_test} correctly classified digits'
                )
            else:
                print(f'Epoch {j} complete')

    def update_batch(self,
                     batch: list,
                     learning_rate: float
    ) -> None:
        '''
        Updates the network's weights and biases by applying stochastic
        gradient descent to the given batch of training observations.
        The gradient is calculated separately in self.backprop().

        args:
            batch (list): A list of tuples (x, y) where x is a (784, 1)
                array representing a training input, and y is a (10, 1)
                array representing a training label.
            learning_rate (float): The learning rate, a number between 0 and 1.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y) -> tuple:
        '''
        For a single training observation, calculates the gradient with
        respect to biases and weights of a quadratic cost function.
        Returns a tuple (nabla_b, nabla_w) representing this gradient.
        This approach is known as backpropagation.

        args:
            x (array): A (784, 1) array representing a training input.
            y (array): A (10, 1) array representing a training label.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data: list) -> int:
        '''
        Uses the neural network to classify observations in the test set
        and return the number of correct predictions.

        args:
            test_data (list): A list of tuples (x, y) representing
                the test data.
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        n_correct = sum(int(x == y) for (x, y) in test_results)

        return n_correct
    
    def cost_derivative(self, output_activations, y):
        '''
        Return vector of partial derivatives of the cost function
        with respect to output activations.

        args:
            output_activations (array): An array of output layer activations.
            y (array): A (10, 1) array representing a training label.
        '''
        derivative = output_activations - y

        return derivative