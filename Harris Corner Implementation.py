import cv2
import numpy as np
from scipy import ndimage as ndi
import os


k = 0.04
window_size = 5
threshold = 10000000.00

corner_detected = 0
edge_detected = 0


input_img = cv2.imread("test_image.png", 0)

cv2.imshow('Input Image', input_img)
cv2.waitKey(0)



corner_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
edge_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)

offset = int(window_size/2)
y_range = input_img.shape[0] - offset
x_range = input_img.shape[1] - offset


dy, dx = np.gradient(input_img)

Ixx = ndi.gaussian_filter(dx**2, sigma=1)
Ixy = ndi.gaussian_filter(dy*dx, sigma=1)
Iyy = ndi.gaussian_filter(dy**2, sigma=1)


for y in range(offset, y_range):
        for x in range(offset, x_range):
            
           
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
           
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            
            r = det - k*(trace**2)

            if r > threshold:
       
                corner_img[y,x] = (0,0,255)
                corner_detected += 1

            elif r < 0:

                edge_img[y,x] = (0,0,255)
    
    
cv2.imshow('Detected Corners', corner_img)
cv2.waitKey(0)
        
cv2.imshow('Detected Edges', edge_img)
cv2.waitKey(0)