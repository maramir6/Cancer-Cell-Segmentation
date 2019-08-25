import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import disk, square
from skimage.filters.rank import enhance_contrast, mean, entropy
from skimage import exposure
from skimage import morphology
from skimage.util import invert
from skimage.morphology import watershed
from skimage.filters import roberts, sobel, scharr, prewitt, median
import cv2
import os
from skimage.filters.rank import equalize


# Image reading
img = io.imread('test11_original.tif')
img_original = img

# Entropy- local variance filter
img_ent = entropy(img, disk(3))
img_norm = cv2.normalize(img_ent,None,0,255,cv2.NORM_MINMAX)
img_norm = np.array(img_norm, dtype=np.uint8)
img_med = median(img_norm, disk(3))
img_med = morphology.remove_small_objects(img_med, 1000)
thresh_ent = threshold_otsu(img_med)
mask_ent = img_med > thresh_ent


# Local gray histogram equalization
img = equalize(img, disk(30), mask=mask_ent)

# Nuclei Segmentation Otsu threshold
img_eq = np.multiply(img, mask_ent)

img = img_original
thresh = threshold_otsu(img_eq)
img_binary = img_eq < 0.5*thresh

img_binary = np.multiply(img_binary, mask_ent)

img_binary = morphology.remove_small_objects(img_binary, 350)
img_binary = morphology.remove_small_holes(img_binary, 100)


# Scharr filter to enhance edge borders 
edge_scharr = 100*scharr(img_eq)
binary_inv = invert(img_binary)
mask_borders = np.multiply(edge_scharr, binary_inv)
mask_borders = np.array(mask_borders, dtype=np.uint8)
mask_borders = exposure.equalize_hist(mask_borders)
mask_borders = invert(mask_borders)
mask_borders = np.array(mask_borders, dtype=np.uint8)


# Label regions and measure properties 

# Watershed Segmentation seeded with nuclei markers
img_borders = cv2.cvtColor(mask_borders,cv2.COLOR_GRAY2RGB)
img_binary = np.array(img_binary, dtype=np.uint8)
ret, markers = cv2.connectedComponents(img_binary)
img_seg = cv2.watershed(img_borders,markers)


# Image windows partition 
lenx, leny = img.shape
window = 256

div_x = int(np.floor(lenx/window))
div_y = int(np.floor(leny/window))

for i in range(0, div_x):
    for j in range(0, div_y):
    	
        x_inf =i*window
        x_sup =(i+1)*window
        y_inf = j*window
        y_sup = (j+1)*window
        
        window_img = img[x_inf:x_sup,y_inf:y_sup]
        window_seg = img_seg[x_inf:x_sup,y_inf:y_sup]
        window_bin = img_binary[x_inf:x_sup,y_inf:y_sup]

        labels = label(window_bin)
        regions = regionprops(labels)
        
        clase = np.array([], dtype=np.uint8)
        #clase = np.array([-1])
        
        for props in regions:
        	y0, x0 = props.centroid
        	clase = np.append(clase,[int(window_seg[int(y0),int(x0)])])

        #clase = np.unique(window_seg)
        #clase = np.delete(clase, 0)
        #print(clase)
        if (j < div_y-1):
            directory =str(i)+str(j)
        
            if not os.path.exists(directory):
                os.makedirs(directory+'/images/')
                os.makedirs(directory+'/masks/')

            file_name = directory+'/images/image_'+str(i)+str(j)+'.png'
            image = np.array(window_img, dtype=np.uint8)
            io.imsave(file_name, image)

            
        
            for k in clase:
                
                mask_image = np.zeros(np.shape(window_img))
                mask_image[window_seg == k] = 1
                mask_image = mask_image > 0.5
                mask_name = directory+'/masks/mask-'+str(k)+'.png'
                io.imsave(mask_name, mask_image)

