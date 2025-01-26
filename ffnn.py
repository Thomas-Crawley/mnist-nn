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

    def sigmoid(z):
        '''
        Applies the sigmoid function to an input z.

        args:
            z (float): A real-valued number.
        '''
        result = 1 / (1 + np.exp(-z))

        return result
    
    def feedforward(self, a):
        '''
        Returns output of the network for the given input.
        
        args:
            a (ndarray): An array of size (n, 1).
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a
    
    def sgd(self,
            training_data,
            n_epochs: int,
            batch_size: int,
            learning_rate: float,
            test_data = None
    ):
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
            learning_rate (float): The learning rate, a number between 0 and 1.
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
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')