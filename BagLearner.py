import numpy as np
import random


class BagLearner(object) :
    
    def __init__(self, learner, kwargs = {"leaf_size":1}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.args = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        
        self.N = 0
        self.learners=[]
        self.leaf_size = 1
        
        if kwargs.has_key("leaf_size"):
            self.leaf_size = kwargs['leaf_size']
        
        
        if(self.bags == 0):
            # If bag count is zero, set it to 1
            self.bags =1
        

        
        for i in range(0,self.bags):
            
            self.learners.append(learner(**kwargs))
            
    
    def author(self):
        return 'pbaldwin6' # replace tb34 with your Georgia Tech username
    
    def addEvidence(self,dataX,dataY):
        self.N = dataX.shape[0]        
        sampled_idx = np.random.choice(range(0,self.N) ,size = self.N, replace=True) 
        trainX = dataX[sampled_idx, :]
        trainY = dataY[sampled_idx]
                
        self.learners[0].addEvidence(trainX, trainY) 

        
        for i in range(1,self.bags):
            # Generate Bag 
            if(self.boost) :
                print "boosting not supported yet"
            else:
                sampled_idx = np.random.choice(range(0,self.N) ,size = self.N, replace=True) 
                trainX = dataX[sampled_idx, : ]
                trainY = dataY[sampled_idx]

                self.learners[i].addEvidence(trainX, trainY)
                

    
    
    def query(self,points):
        
        bag_predY = np.zeros(points.shape[0])
        for i in range(self.bags):
            predY = self.learners[i].query(points)
            bag_predY += predY
        
        bag_predY = bag_predY/self.bags
        
        return(bag_predY)
        
        
