'''
Train a neural network and print the results.
'''

import ffnn
import mnist_loader

# Load data
training_data, validation_data, test_data = mnist_loader.load_data('mnist.pkl.gz')

# Initialise network with 30 neurons in the hidden layer
network = ffnn.FFNN([784, 30, 10])

# Train the network and print the test set results for each epoch
network.sgd(training_data = training_data,
            n_epochs = 30,
            batch_size = 10,
            learning_rate = 3.0,
            test_data = test_data)