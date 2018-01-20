"""Implement Insane Learner"""

import numpy as np
import LinRegLearner, DTLearner, RTLearner, BagLearner


class InsaneLearner(object):

    def __init__(self, bag_learner=BagLearner.BagLearner, learner=DTLearner.DTLearner, 
        num_bag_learners=20, verbose=False, **kwargs):
        """Initalize an Insane Learner

        Parameters:
        bag_learner: A BagLearner
        learner: A LinRegLearner, DTLearner, or RTLearner to be called by bag_learner
        num_bag_learners: The number of Bag learners to be trained
        verbose: If True, information about the learner will be printed out
        kwargs: Keyword arguments to be passed on to the learner's constructor
        
        Returns: An instance of Insane Learner
        """
        self.verbose = verbose
        bag_learners = []
        for i in range(num_bag_learners):
            bag_learners.append(bag_learner(learner=learner, **kwargs))
        self.bag_learners = bag_learners
        self.kwargs = kwargs
        self.num_bag_learners = num_bag_learners
        if verbose:
            self.get_learner_info()

        
    def addEvidence(self, dataX, dataY):
        """Add training data to learner

        Parameters:
        dataX: A numpy ndarray of X values to add
        dataY: A numpy 1D array of Y values to add

        Returns: Updated individual bag learners within InsaneLearner
        """
        for bag_learner in self.bag_learners:
            bag_learner.addEvidence(dataX, dataY)
        if self.verbose:
            self.get_learner_info()
        
        
    def query(self, points):
        """Estimates a set of test points given the model we built
        
        Parameters:
        points: A numpy ndarray of test queries

        Returns: 
        preds: A numpy 1D array of the estimated values
        """
        preds = np.array([learner.query(points) for learner in self.bag_learners])
        return np.mean(preds, axis=0)


    def get_learner_info(self):
        """Print out data for this InsaneLearner"""
        bag_learner_name = str(type(self.bag_learners[0]))[8:-2]
        print ("This InsaneLearner is made up of {} {}:".
            format(self.num_bag_learners, bag_learner_name))
        print ("kwargs =", self.kwargs)

        # Print out information for each learner within InsaneLearner
        for i in range(1, self.num_bag_learners + 1):
            print (bag_learner_name, "#{}:".format(i)); 
            self.bag_learners[i-1].get_learner_info() 


if __name__=="__main__":
    print ("This is a Insane Learner\n")
    
    # Some data to test the InsaneLearner
    x0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
    x1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
    x2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
    x = np.array([x0, x1, x2]).T
    
    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create an InsaneLearner from given training x and y
    insane_learner = InsaneLearner(verbose=True)
    
    print ("\nAdd data")
    insane_learner.addEvidence(x, y)
    
    # Query with dummy data
    print ("Query with dummy data:\n", np.array([[1, 2, 3], [0.2, 12, 12]]))
    print ("Query results:", insane_learner.query(np.array([[1, 2, 3], [0.2, 12, 12]]))) 
    
