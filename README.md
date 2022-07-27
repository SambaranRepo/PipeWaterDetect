# PipeWaterDetect
This script is used to detect standing water in from images collected inside pipelines using the pipeline inspection robot. 
This method comprises of three steps : 
1. Using a <pre>roipoly</pre> tool, we select a region of pixels first belonging to the water class and second belonging to the non water class. This step is done over the collection of images and comprises of our training data to train the classifiers. The collected pixels are saved into a pickle file for further use.
2. Using the pixel data collected for the water and non water classes, we train a gaussian classifier for each class and save the distribution parameters in a pickle file for further use. 
3. Now given a pipeline image, the image is first segmented into a binary mask image according to our gaussian classifiers for the water and non water class. Given the segmneted image, we then make use of cv2 contour methods and some shape statistics to filter out regions that possibly indicate water region. 

## Data Set
1. Make a directory named ```WaterData``` and place all files that contain images of pipes containing standing water into this folder. 
2. Make a directory named ```NonWaterData``` and place all files that contain images of pipes not containing any standing water into this folder. 

## Selecting region of interest pixels from the images 
We first run the ```generate_color_data.py``` script to select pixels of water and non water regions from the images in our dataset. First select the color space you would like to do the classification in such as RGB, YCrCb, HSV etc. For this, just change the last 3 characters in line 38 of this script such that it reads something like ```img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)``` or ```img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)```. 

For each image, first we have to select the region containing water. To start creating the polygon which will contain our interested pixels, left click mouse at the starting point, extend the mouse to make a straight line, left click again at a point where you would like to make the second vertex, and keep repeating until you make a closed polygon region. When done, right click mouse and the image will close, and all pixels inside the polygon created will get assigned as water pixels. 

The same image will open again, repeat the same process above but this time for non water region. The polygon in this case will be green in color contrary to the red polygon in the previous case. 

Finally, the pixels are saved into a pickle file. Pickle files are named according to the color space we selected the pixels in. For ex : ```WaterClassifier.pkl```, ```WaterClassifierYCrCb.pkl``` or ```WaterClassifierHSV.pkl```

Usage : 
```python3 generate_color_data.py```

## Training the gaussian classifier using the pixels in the desired color space
After we have collected the pixels for the water and non water classes in a desired color space, we can load these pixels from the pickle file, and then we train a gaussian classifier for each of the water and non water pixels. The distribution parameters are then saved into another pickle file with naming according to the color space the pixels, such as ```water_classifier_model_ycrcb.pkl``` or ```water_classifier_hsv.pkl```

Usage : 
```python3 gaussian_classifier.py```

## Using the classifier to segment the image and detecting standing water
Afer we have the trained gaussian classifier, given an image of the pipe, we segment the image into water like pixels and non water like pixels. Water like pixels are colored white and non water like pixels are colored black. The ```WaterDetector``` class contains the function ```segment_image``` that takes as input the original image and does this segmentation. All we have to do is load the desired gaussian classifier model parameters and correspondingly change the color space of the image. For ex, if we load the gaussian parameters that were trained in YCrCb color space, change line 30 to read ```with open('water_classifier_model_ycrcb.pkl' ,'rb') as f:``` and change line 43 to read ```img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)``` and similar for other color spaces. 

Based on the segmentation masks, we use cv2 contour functions and some shape statistics to filter out regions that are possibly standing water. The ```bounding_box``` function inside the ```WaterDetector``` class does this by taking as input the masked image provided by ```segment_image```. 

To use this script, we need to give an extra argument : Provide arg ```Water``` to deploy the algorithm on the images containing water which loads images from ```WaterData``` directory, or ```NonWater``` which loads images from ```NonWaterData``` directory. 

Usage : 
```python3 water_classifier.py Water```
or 
```python3 water_classifier.py NonWater```

## Testing 
In line 

## ToDo (25th July, 2022): 
1. Train gaussian classifier with more data 
2. See cv2 erosion and dilation to remove dark pixels between black pixels
3. More tuning of h/w and area_ratio to get proper bounding boxes.
4. See if converting to other color spaces such as YUV, HCrCb helps in segmentation 

## ToDo (26th July, 2022):
1. using cv2 erode and dilate (tune number of iterations / kernel size)
2. YcrCb space currently giving best performance : 
    details : erode : kernel 7,7 iterations 3
              dilate : kernel 5,5 iterations 1

              False positives : 15 
              False negatives : 0
3. See if better segmentation can be achieved either via collecting more data or any other color space. If not see what tuning in bounding box function gets best performance. 


    ### Best performance till now
   erode : kernel 7,7 iterations 7 
   dilation : kernel 5,5 iterations 1
   False negatives : 10 (although for the false negative next images have water detected in them)
   False positives : 7

## Results

### Performance in standing water pipe 
 
<p align='center'>
<img src="./gif/water.gif">
</p>

### Performance in clear pipe 
 
<p align='center'>
<img src="./gif/nonwater.gif">
</p>