'''
train a gaussian classifier for water and non water
water class will have associated gaussian distributions
non water class will have another gaussian distribution
required params : mu, cov, prior
'''

import numpy as np
import pickle

class GaussianClassifier:
    '''
    train gaussian classifier using provided data
    '''

    def __init__(self, x_water, x_non_water):
        '''
        initialise data
        '''

        self.x_water = x_water
        self.x_non_water = x_non_water
        self.total = len(self.x_water) + len(self.x_non_water)

    def estimate_prior(self, x):
        '''
        determine prior probability of a class
        prior = number of sample of class / total samples
        '''

        return len(x) / self.total

    def estimate_mean(self, x):
        '''
        estimate mean value of given class 
        mu = 1/N * sum(x)
        our data is image pixels, so we want a mean pixel
        hence mean along row
        '''

        return np.mean(x, axis = 0)

    def estimate_cov(self, x):
        '''
        estimate the covariance of the sample data
        cov = 1/ N xx.T
        image pixel is of 1*3, mean is 1*3, so cov should be 3*3
        x is N * 3, so xx.T will give N*N, we need x.T x
        '''

        return np.cov(x.T)

    def gaussian_classifier(self):
        '''
        estimate the parameters of the water and non water gaussian classifiers
        Let y = 1 represent water and y = 0 represent non water
        two sets of parameters : For y = 1 {mu_1, sigma_1, p_1} 
        For y = 0 {mu_0, sigma_0, p_0}
        '''

        p_1 = self.estimate_prior(self.x_water)
        p_0 = self.estimate_prior(self.x_non_water)

        mu_1 = self.estimate_mean(self.x_water)
        mu_0 = self.estimate_mean(self.x_non_water)

        sigma_1 = self.estimate_cov(self.x_water)
        sigma_0 = self.estimate_cov(self.x_non_water)

        params = [p_1, p_0, mu_1, mu_0, sigma_1, sigma_0]

        return params


if __name__ == '__main__':
    with open('WaterClassifier.pkl', 'rb') as f:
        x = pickle.load(f)
        x_water, x_non_water = x[0], x[1]

    water_classifier = GaussianClassifier(x_water, x_non_water)
    params = water_classifier.gaussian_classifier()

    with open('water_classifier_model.pkl', 'wb') as f:
        pickle.dump(params, f)
        
