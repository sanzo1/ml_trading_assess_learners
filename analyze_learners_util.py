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


def train_test_learner(learner_arg, num_iterations=1, max_leaf_size=None, 
    max_bag_size=None, **kwargs):
    
    """Train and test a learner

    Parameters:
    learner_arg: A DTLearner, RTLearner or BagLearner
    num_iterations: Number of times we train and test the data
    max_leaf_size: The max value of the leaf size range on which we train a tree learner
    max_bag_size: The max value of the bag size range on which we train a bag learner
    kwargs: Keyword arguments to be passed on to the learner's constructor
    
    Returns:
    RMSEin_mean: A numpy 1D array of means of root mean square errors (RMSEs) 
                for in sample data
    RMSEout_mean: A numpy 1D array of means of RMSEs for out of sample data
    CORRin_mean: A numpy 1D array of medians of correlations 
                    between predicted and actual results for in sample data
    CORRout_mean: A numpy 1D array of medians of correlations for out of sample data
    """

    # Make sure that either of these variables is not None
    if max_leaf_size is None and max_bag_size is None:
        print ("Please specify the max_leaf_size or max_bag_size and try again;")
        print ("Returning fake data filled with zeros for now")
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

    max_val = max_leaf_size or max_bag_size
    # Initialize two ndarrays for in sample and out of sample root mean squared errors
    RMSEin = np.zeros((max_val, num_iterations))
    RMSEout = np.zeros((max_val, num_iterations))

    # Initialize two ndarrays for in sample and out of sample correlations
    CORRin = np.zeros((max_val, num_iterations))
    CORRout = np.zeros((max_val, num_iterations))

    # Train the learner and record RMSEs
    for i in range(max_val):
        for j in range(num_iterations):
            # Create a learner and train it
            learner = learner_arg(leaf_size=i, **kwargs)
            learner.addEvidence(trainX, trainY)

            # Evaluate in sample
            predY = learner.query(trainX)
            rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            RMSEin[i, j] = rmse
            c = np.corrcoef(predY, y=trainY)
            CORRin[i, j] = c[0, 1]

            # Evaluate out of sample
            predY = learner.query(testX)
            rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            RMSEout[i, j] = rmse
            c = np.corrcoef(predY, y=testY)
            CORRout[i, j] = c[0, 1]
    
    # Get the means of RMSEs from all iterations
    RMSEin_mean = np.mean(RMSEin, axis=1)
    RMSEout_mean = np.mean(RMSEout, axis=1)

    # Get the medians of correlations from all iterations
    CORRin_mean = np.median(CORRin, axis=1)
    CORRout_mean = np.median(CORRout, axis=1)

    return RMSEin_mean, RMSEout_mean, CORRin_mean, CORRout_mean


def plot_results(in_sample, out_of_sample, title, xlabel, ylabel, 
    legend_loc="lower right", xaxis_length=1):
    
    """Plot the results, e.g. RMSEs or correlations from training and testing a learner
    
    Parameters:
    in_sample: A numpy 1D array of in sample data
    out_of_sample: A numpy 1D array of out of sample data
    title: The chart title
    xlabel: x-axis label
    ylabel: y-axis label
    legend_loc: Location of legend
    xaxis_length: The length of the x-axis

    Returns: Plot the data
    """

    xaxis = np.arange(1, xaxis_length + 1)
    plt.plot(xaxis, in_sample, label="in sample", linewidth=2.0)
    plt.plot(xaxis, out_of_sample, label="out of sample", linewidth=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.title(title)
    plt.show()