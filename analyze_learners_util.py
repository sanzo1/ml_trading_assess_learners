"""Utils for analyzing learners"""

import numpy as np
import math
import matplotlib.pyplot as plt


def process_data(filename, train_size=0.6):
    """Reads data from a file and split the data into training and test sets"""
    data = np.genfromtxt(filename, delimiter=",")
    
    # If the data has a header, remove it
    if np.isnan(data[0]).all():
        print ("Remove the header")
        data = data[1:]
    
    # If the data's first column is non-numerical (e.g. date), remove it
    if np.isnan(data[:, 0]).all():
        print ("Remove the non-numerical column (1st one)")
        data = data[:, 1:]

    np.random.shuffle(data)

    # Compute how much of the data is training and testing
    train_rows = int(math.floor(train_size * data.shape[0]))
    test_rows = data.shape[0] - train_rows

    # Separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    return trainX, trainY, testX, testY


