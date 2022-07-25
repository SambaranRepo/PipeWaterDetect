from email import iterators
import numpy as np
import cv2
import pickle
from math import pi
from glob import glob
from matplotlib import pyplot as plt
import sys,os

class WaterDetector:
    '''
    class to first segment the image according to the water and non water classses
    Then use cv2 contour methods to find appropriate regions of water
    '''

    def __init__(self):
        '''
        initialise the water and non water gaussian parameters
        '''
        with open('water_classifier_model.pkl' ,'rb') as f:
            params = pickle.load(f)

        self.prior_1, self.prior_0, self.mu_1, self.mu_0, self.sigma_1, self.sigma_0 = params

        self.mu_1 = self.mu_1[:,None].T
        self.mu_0 = self.mu_0[:,None].T

    def segment_image(self, img):
        '''
        segment the image into water and non water
        segmented image is a binary image with white representing water and black representing non water
        '''

        x = img.reshape(img.shape[0] * img.shape[1] , img.shape[2])
        mask_img = np.zeros(img.shape[0] * img.shape[1], dtype = np.uint8)

        step = 100
        K = len(x) // step

        for i in range(K):
            water_likelihood = self.gaussian_posterior_likelihood(x[step * i : step * (i + 1)], self.mu_1, self.sigma_1, self.prior_1)
            non_water_likelihood = self.gaussian_posterior_likelihood(x[step * i : step * (i + 1)], self.mu_0, self.sigma_0, self.prior_0)
            mask_img[step*i : step*(i + 1)] = water_likelihood < non_water_likelihood
        
        water_likelihood = self.gaussian_posterior_likelihood(x[step * K : len(x)], self.mu_1, self.sigma_1, self.prior_1)
        non_water_likelihood = self.gaussian_posterior_likelihood(x[step * K : len(x)], self.mu_0, self.sigma_0, self.prior_0)
        mask_img[step * K : len(x)] = water_likelihood < non_water_likelihood

        mask_img = mask_img.reshape(img.shape[0], img.shape[1])

        return mask_img

    def bounding_box(self, mask_img):
        '''
        using the segmented image, try to get a bounding box over the water region
        using cv2 contour methods and shape statistics
        '''
        mask = mask_img
        x_max, y_max = mask.shape[0], mask.shape[1]

        mask *= 255 
        kernel = np.ones((11,11), np.uint8)
        erode = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erode, kernel[:5, :5], iterations = 5)
        blurred = cv2.GaussianBlur(dilation, (5,5), 0)
        ret, thresh = cv2.threshold(blurred, 127, 255, 0)

        boxes = []
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x,y,w,h = cv2.boundingRect(cnt)
            area_ratio = cv2.contourArea(cnt) / (x_max * y_max)
            print(f'area ratio : {area_ratio}')
            print(f'height / width : {h/w}')
            
            if 0.5 <= h/w <= 1.5 and 0.008 <= area_ratio<=0.15:
                boxes.append([x,y,x + w, y + h])
                cv2.rectangle(img, (x,y), (x + w,y + h), (255,0,0), 10)
                print(f'area ratio : {area_ratio}')
                print(f'height / width : {h/w}')
                fig,ax = plt.subplots()
                ax.imshow(img)
                plt.show(block = True)
        
        boxes.sort()
        return boxes

    def gaussian_posterior_likelihood(self, X, mu, cov, prior): 
        '''
        : Compute the log of the probability of class given X
        : output: log(P(X|class)) + log(P(class)) = (X - mu)^T * cov^(-1) * (X - mu) + log(det(cov)) - 2*log(P(class))
        '''
        return np.diag(((X - mu).dot(np.linalg.inv(cov)).dot((X - mu).T))) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)


if __name__ == '__main__':
    folder = glob('WaterData/')[0]
    files = os.listdir(folder)

    for i in range(10):
        filename = files[i]
        img = cv2.imread(os.path.join(folder + filename))

        water_detector = WaterDetector()
        mask = water_detector.segment_image(img)

        fig,ax = plt.subplots()
        ax.imshow(~mask, cmap = plt.cm.binary)
        ax.axis('off')
        plt.show(block = True)

        boxes = water_detector.bounding_box(mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in boxes:
            x1,y1,x2,y2 = box
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 10)
        
        fig,ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        plt.show(block = True)