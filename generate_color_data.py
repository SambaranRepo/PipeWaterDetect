'''
script to select water and not water regions inside the pipe using roipoly
'''

import os 
import cv2
import numpy as np
import pickle
from glob import glob 
from matplotlib import pyplot as plt
from roipoly import RoiPoly, MultiRoi
from tqdm import tqdm

class GenerateData:
    '''
    class to generate different class data from samples
    '''

    def __init__(self):
        pass

    def generate_color_data(self, folder):
        '''
        given the rgb images in the folder, open them
        use roipoly to select water regions inside the pipe
        and then select non water regions 
        '''

        n = len(list(os.listdir(folder)))
        x_water = np.empty([1,3], dtype = np.int32)
        x_non_water = np.empty([1,3], dtype = np.int32)
        files = os.listdir(folder)

        # for i in range(len(files)):
        for i in tqdm(range(0,10)):
            file = files[i]
            img = cv2.imread(os.path.join(folder + file))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

            fig,ax = plt.subplots()
            ax.imshow(img)

            my_roi_1 = RoiPoly(fig = fig, ax = ax, color = 'r')
            mask = np.asarray(my_roi_1.get_mask(img))
            x_water = np.concatenate([x_water, img[mask == 1]], axis = 0)

            fig,ax = plt.subplots()
            ax.imshow(img)
            my_roi_2 = RoiPoly(fig = fig, ax = ax, color = 'g')
            mask = np.asarray(my_roi_2.get_mask(img))
            x_non_water = np.concatenate([x_non_water, img[mask == 1]], axis = 0)
        
        return x_water, x_non_water

if __name__ == '__main__':
    folder = glob('WaterData/')[0]
    print(f'folder : {folder}')
    data_gen = GenerateData()

    with open('WaterClassifierLab.pkl', 'ab') as f:
        x_water, x_non_water = data_gen.generate_color_data(folder)
        pickle.dump([x_water, x_non_water], f)
