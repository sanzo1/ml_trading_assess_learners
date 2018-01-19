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

        
