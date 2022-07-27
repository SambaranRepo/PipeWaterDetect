# PipeWaterDetect
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