'''
Functions to load the MNIST dataset.
'''

# Import packages
import gzip
import numpy as np
import pickle

# Define a function to vectorise digits
def vectorise_digit(j):
    '''
    Given a digit 0-9 as input, returns a vector with 10 elements,
    with 1.0 in the j-th element and 0.0 in all other elements.
    This will be the output format of the neural network.
    '''

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Define a function to load MNIST dataset
def load_data(file):
    '''
    Load the MNIST dataset from a zipped pickle file.
    Returns a tuple containing (train_data, val_data, test_data).
    '''

    # Load raw data
    with gzip.open(file, 'rb') as f:
        train_raw, val_raw, test_raw = pickle.load(f, encoding = 'latin1')

    # Create training data tuple
    train_X = [np.reshape(x, (784, 1)) for x in train_raw[0]]
    train_y = [vectorise_digit(y) for y in train_raw[1]]
    train_data = list(zip(train_X, train_y))

    # Create validation data tuple (no need to vectorise Y)
    val_X = [np.reshape(x, (784, 1)) for x in val_raw[0]]
    val_y = val_raw[1]
    val_data = list(zip(val_X, val_y))

    # Create test data tuple (no need to vectorise Y)
    test_X = [np.reshape(x, (784, 1)) for x in test_raw[0]]
    test_y = test_raw[1]
    test_data = list(zip(test_X, test_y))

    return(train_data, val_data, test_data)