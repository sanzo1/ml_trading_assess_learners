"""A wrapper for Bag Learner"""

import numpy as np
import pandas as pd
from copy import deepcopy
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

        
    