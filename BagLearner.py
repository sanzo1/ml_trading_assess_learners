"""Implement Bag Learner"""
# TODO: Implement Boosting

import numpy as np
import LinRegLearner, DTLearner, RTLearner


class BagLearner(object):

    def __init__(self, learner, bags=20, boost=False, verbose=False, **kwargs):
        """Initalize a Bag Learner

        Parameters:
        learner: A LinRegLearner, DTLearner, or RTLearner
        bags: The number of learners to be trained using Bootstrap Aggregation
        boost: If true, boosting will be implemented
        verbose: If True, information about the learner will be printed out
        kwargs: Keyword arguments to be passed on to the learner's constructor
        
        Returns: An instance of Bag Learner
        """
        self.verbose = verbose
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        if verbose:
            self.get_learner_info()

        
    def addEvidence(self, dataX, dataY):
        """Add training data to learner

        Parameters:
        dataX: A numpy ndarray of X values to add
        dataY: A numpy 1D array of Y values to add

        Returns: Updated individual learners in BagLearner
        """
        # Sample the data with replacement
        num_samples = dataX.shape[0]
        for learner in self.learners:
            idx = np.random.choice(num_samples, num_samples)
            bagX = dataX[idx]
            bagY = dataY[idx]
            learner.addEvidence(bagX, bagY)
        if self.verbose:
            self.get_learner_info()
        
        
    def query(self, points):
        """Estimates a set of test points given the model we built
        
        Parameters:
        points: A numpy ndarray of test queries

        Returns: 
        preds: A numpy 1D array of the estimated values
        """
        preds = np.array([learner.query(points) for learner in self.learners])
        return np.mean(preds, axis=0)


    def get_learner_info(self):
        """Print out data for this BagLearner"""
        learner_name = str(type(self.learners[0]))[8:-2]
        print ("This BagLearner is made up of {} {}:".
            format(self.bags, learner_name))

        print ("kwargs =", self.kwargs)
        print ("boost =", self.boost)

        # Print out information for each learner within BagLearner
        for i in range(1, self.bags + 1):
            print (learner_name, "#{}:".format(i)); 
            self.learners[i-1].get_learner_info() 


