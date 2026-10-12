import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os 
import pandas as pd

def machinevisionstuff(imagedraw,scalefactor,tomswindow): # returns volume in mm3 or micro L 
    
    TWgap = (1- tomswindow)/2
    width, height = imagedraw.size

    # get ranges for tom's window 
    TW_Xmax =  width * (1- TWgap)
    TW_Xmin =  width * (TWgap)

    TW_Ymax =  height * (1- TWgap)
    TW_Ymin =  height * (TWgap)
    
    
    here = os.getcwd()
    # Set a threshold value
    threshold_value = 128  # Adjust this value based on your specific image characteristics
    image_array = np.array(imagedraw)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert binary image to single-channel (grayscale)
    grayscale_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    # Find connected components in the grayscale image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grayscale_image, connectivity=8)

    # Lists to store highest and lowest points for each blob
    top_points_list = []
    bot_points_list = []
    left_points_list = []
    right_points_list = []
    rects = []
    # Iterate through each blob
    for label in range(1, num_labels):  # Skip the background label 0
        # Extract the region of interest (ROI) for the current blob
        roi = (labels == label).astype(np.uint8)
        # Find the coordinates of the non-zero pixels in the blob
        non_zero_points = np.column_stack(np.where(roi > 0))

        # Find the highest and lowest points within the blob
        top_point = np.min(non_zero_points[:,0], axis=0)
        bot_point = np.max(non_zero_points[:,0], axis=0)

        left_point = np.min(non_zero_points[:,1], axis=0)
        right_point = np.max(non_zero_points[:,1], axis=0)
        
        
        startX = left_point
        endX = right_point
        
        startY = top_point
        endY = bot_point
        
        midX = (startX + endX)/2
        midY = (startY + endY)/2
        
        if TW_Xmin< midX <TW_Xmax :
            if TW_Ymin< midY <TW_Ymax :
    
                # Append the points to the lists
                top_points_list.append(top_point)
                bot_points_list.append(bot_point)

                left_points_list.append(left_point)
                right_points_list.append(right_point)

    # Convert lists to arrays if needed
    top_points_array = np.array(top_points_list)
    bot_points_array = np.array(bot_points_list)

    left_points_array = np.array(left_points_list)
    right_points_array = np.array(right_points_list)

    # add stuff into tracker object called objects because it an object that stores info of tracked objects


    # make variables

    pi5 = 3.1415
    radiset1 = right_points_array - left_points_array # in pixel length
    radiset2 = bot_points_array - top_points_array # in pixel length


    center = [ (bot_points_array + top_points_array)/2,(right_points_array + left_points_array)/2]
    Y_data = (bot_points_array + top_points_array)/2
    X_data = (right_points_array + left_points_array)/2
    radiset1 = radiset1 / scalefactor  # in mm length 
    radiset2 = radiset2 / scalefactor  # in mm length 

    meanradi = (radiset1 + radiset2) /2 /2 # 2nd /2 is to convert from diameter to radi
    hemivolume = 2/3*pi5*(meanradi*meanradi*meanradi) # no idea how to do ^3 
    spherevolume = hemivolume * 2
    numberofbubbles = len(hemivolume)

    volumedata = hemivolume.tolist()

    return volumedata , X_data , Y_data, meanradi ,radiset1 , radiset2