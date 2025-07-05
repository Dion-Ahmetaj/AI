import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def StochasticAverageGradient (x, y, theta, alpha, num_iters, c):

        # get the number of samples in the training
        m = x.shape[0]
        

        for i in range(0, num_iters):
            
            # find linear regression equation value, X and theta
            z = np.dot(x, theta)
            
            # get the sigmoid of z
            h = sigmoid(z)
            
            # let's add L2 regularization
            # c is L2 regularizer term
            likelyhood= (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))) 
            
        
            
            # update the weights theta
            theta = (1 - 2 * c * alpha) * theta + (alpha * 1/m * np.dot(x.T, (y - h)))
    
        likelyhood = float(likelyhood)
        return likelyhood, theta

class logisticregression():

    def __init__(self, n_iters=100,theta=None,j=None,c=1e-7,alpha=0.7,size_of_voc=1):
        self.n_iters = n_iters
        #weights
        self.theta = np.zeros((size_of_voc, 1))
        #loss
        self.j = j
        #λ 
        self.c=c
        self.alpha=alpha
        self.size_of_voc=size_of_voc
        self.weights=None
        

    

    def fit(self, x, y):
        np.random.seed(1)
        # Apply gradient descent of logistic regression
        # 0.1 as added L2 regularization term
        likelyhood, theta= StochasticAverageGradient(x, np.array(y).reshape(-1,1), np.zeros((self.size_of_voc, 1)), self.alpha, self.n_iters, self.c)
        #print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
        self.weights=theta
        
        


    def predict(self, x,theta):
        y_pred = sigmoid(np.dot(x,theta))
        return y_pred
    

    