import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os 
import pandas as pd
import math

def Create_bubble_IDs(df_xpos,df_ypos,df_ID,df_xrange,df_yrange,imagenumber,numberofbubbles, width, height, higestIDnumber):
    highestscore = 0
    downscalefactor = 4 
    def jaccard_similarity_score(arr1, arr2):
        intersection = np.logical_and(arr1, arr2)
        union = np.logical_or(arr1, arr2)
        return intersection.sum() / union.sum()
    
    
    
    pastcolumnnames = df_xpos.columns.tolist()
    columnname = "time = "+str(imagenumber)
    xpos_series = df_xpos[columnname]
    ypos_series = df_ypos[columnname]
    
    xrange_series = df_xrange[columnname]
    yrange_series = df_yrange[columnname]
    
    ID_series  = xpos_series * 0 # make blank series for IDs    
    numberofbubbles = xpos_series.notna().sum()

    j=0
    entry_number = 0

    while entry_number < numberofbubbles : 

        ID_entry = 0 
        jaccard_score = -1 
        # create numpy array box 
        #print(entry_number)
        entry_array = np.zeros((int(height/downscalefactor),int(width/downscalefactor))) # numpy just does it this way
        X1 = xpos_series[entry_number] - xrange_series[entry_number]/2
        Y1 = ypos_series[entry_number] - yrange_series[entry_number]/2
        X2 = xpos_series[entry_number] + xrange_series[entry_number]/2
        Y2 = ypos_series[entry_number] + yrange_series[entry_number]/2
        Top_left   = (int(Y1/downscalefactor) , int(X1/downscalefactor))
        Bot_right  = (int(Y2/downscalefactor) , int(X2/downscalefactor))
        entry_Center= ( (int(X1) - int(X2))/2 , (int(Y1) - int(Y2))/2 )
        #print(Top_left , Bot_right)
        entry_array[Top_left[0]:Bot_right[0], Top_left[1]:Bot_right[1]] = 255               

        if len(pastcolumnnames) == 1 :
            # skip looking into past as 
            ID_entry = higestIDnumber + 1 
            higestIDnumber = higestIDnumber + 1 
            #print(ID_entry)
        else :

            # shift thorugh past IDs 
            past_columnnumber = 2 # start at 2 as -1 is last entry 
            while past_columnnumber < 4: 

                # grab from past column 
                past_column_name     = pastcolumnnames[-past_columnnumber]


                if past_column_name == "Have 255 you have no choice" :
                    past_columnnumber = 10000
                    break

                past_xpos_series     = df_xpos[past_column_name]
                past_ypos_series     = df_ypos[past_column_name]

                past_xrange_series   = df_xrange[past_column_name]
                past_yrange_series   = df_yrange[past_column_name]
                past_ID_series       = df_ID[past_column_name]
                past_numberofbubbles = past_xpos_series.notna().sum()

                #print(past_xpos_series)
                #print(past_numberofbubbles)
                grab_number = 0 
                highestscore = 0
                while grab_number < past_numberofbubbles : 
                    
                    
                    
                    past_ID_entry = past_ID_series[grab_number]
                    # create numpy array box for past IDed bubbles
                    past_X1 = past_xpos_series[grab_number] - past_xrange_series[grab_number]/2
                    past_Y1 = past_ypos_series[grab_number] - past_yrange_series[grab_number]/2
                    past_X2 = past_xpos_series[grab_number] + past_xrange_series[grab_number]/2
                    past_Y2 = past_ypos_series[grab_number] + past_yrange_series[grab_number]/2
                    
                    # Check if any of the variables contain NaN
                    if any(np.isnan(var) for var in [past_X1, past_Y1, past_X2, past_Y2]):
                        print("we got NaN values in past box values")
                        print(grab_number)
                        print(past_numberofbubbles)
                        print(past_X1, past_Y1, past_X2, past_Y2)
                        print(past_xpos_series.head(10))
                        break

                    
                    past_Top_left   = (int(past_Y1/downscalefactor) , int(past_X1/downscalefactor))
                    past_Bot_right  = (int(past_Y2/downscalefactor) , int(past_X2/downscalefactor))
                    past__Center= ( (int(past_X1) - int(past_X2))/2 , (int(past_Y1) - int(past_Y2))/2 )
                    
                    Gagx = past__Center[0] - entry_Center[0]
                    Gagy = past__Center[1] - entry_Center[1]
                    
                    Gap = (Gagx*Gagx + Gagy*Gagy)
                    Gap = math.sqrt(Gap)
                    
                    Bubble_diam = (past_X2 - past_X1)/2 + (past_Y2 - past_Y1)/2 
                    
                    if Gap < Bubble_diam*1.5 :
                        # grab past ID bubbles to check if any exist
                        grabbed_array = np.zeros((int(height/downscalefactor),int(width/downscalefactor))) # numpy just does it this way
                        
                        grabbed_array[past_Top_left[0]+0:past_Bot_right[0]+0, past_Top_left[1]+0:past_Bot_right[1]+0] = 255  

                        if jaccard_score == - 1 :
                            #first test 
                            highestscore = jaccard_similarity_score(entry_array, grabbed_array)
                            higestID = past_ID_entry


                        jaccard_score = jaccard_similarity_score(entry_array, grabbed_array)

                        if jaccard_score > highestscore : 
                            highestscore = jaccard_score
                            higestID = past_ID_entry
                    

                    grab_number = grab_number + 1 
                if highestscore > 0.5 :
                    ID_entry = higestID
                    #print("ID given by test")
                past_columnnumber = past_columnnumber + 1 

        # shove ID_entry into Id_seies 
        if ID_entry == 0 :
            ID_entry = higestIDnumber + 1 
            higestIDnumber = higestIDnumber + 1 
            #print("no ID match found")

        ID_series[entry_number] = ID_entry
        #print(ID_entry)

        entry_number = entry_number + 1

    df_ID[columnname] = ID_series
        
        
    return df_ID , higestIDnumber