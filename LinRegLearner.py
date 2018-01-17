"""A wrapper for linear regression"""

import numpy as np


class LinRegLearner(object):

    def __init__(self, verbose = False):
        """Initalize a Linear Regression Learner

        Parameters:
        model_coefs: A nummpy ndarray of least-squares solution
        residuals: A numpy ndarray of sums of residuals
        rank: The rank of the first input matrix to np.linalg.lstsq()
        s: An ndarray of singular values of the first input matrix to np.linalg.lstsq()

        Returns: An instance of Linear Regression Learner
        """
        self.model_coefs = None
        self.residuals = None
        self.rank = None
        self.s = None
        self.verbose = verbose
        if verbose:
            self.get_learner_info()


    def addEvidence(self, dataX, dataY):
        """Add training data to learner

        Parameters:
        dataX: A numpy ndarray of X values of data to add
        dataY: A numpy 1D array of Y values to add

        Returns: Update the instance variables
        """

        # Add 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:,0:dataX.shape[1]] = dataX

        # build and save the model
        self.model_coefs, self.residuals, self.rank, self.s = np.linalg.lstsq(newdataX, dataY)

        if self.verbose:
            self.get_learner_info()
        
    
    def query(self, points):
        """Estimate a set of test points given the model we built

        Parameters:
        points: A numpy array with each row corresponding to a specific query

        Returns: the estimated values according to the saved model
        """
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]


    def get_learner_info(self):
        """Print out data for this learner"""
        print ("Model coefficient matrix:", self.model_coefs)
        print ("Sums of residuals:", self.residuals)
        print ("")


if __name__=="__main__":
    print ("This is a Linear Regression Learner")
