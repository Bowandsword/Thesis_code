# import my functions 
from functionstorage.Create_bubble_ID import *
from functionstorage.MachineVisionFunction import *
from functionstorage.ProccessFunction3 import *
from functionstorage.SetupDataframeFunction import *
from functionstorage.WritetoDataframesFunction import *

# General stuff
import time
import os 
here = os.getcwd()
import random
import shutil
import math
import pandas as pd
import func_timeout
import sys
import tkinter as tk

# image stuff
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output
from IPython.display import Image  # for displaying images
import matplotlib.pyplot as plt

# AI stuff
from ultralytics import YOLO
import torch
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from pathlib import Path
from fastai.vision.all import *

# -------------------------------------importing ends -----------------------------------


# ------------------------------------ collect user inputs via GUI ---------------------------
# ----------------- GUI Start ---------------#

# -----------------get parent folder locations --------------#

def save_input():
    global user_input
    user_input = entry.get()
    root.destroy()
root = tk.Tk()
root.title("String Input")
root.geometry("400x200")  # Set window size to 300x100 pixels
# Define a font with a larger size
font = ("Helvetica", 24)
label = tk.Label(root, text="Input path to parent folder:", font=font)
label.pack()
entry = tk.Entry(root, font=font)
entry.pack()
button = tk.Button(root, text="Next", command=save_input, font=font)
button.pack()
root.mainloop()

if user_input == "" :
    # use default
    s3dir =  "C:/Users/bowan/Pictures/geliondata_local/magnet_side/"

else :
    
    s3dir = user_input 

spreadsheetdir = s3dir+"/cyclingdata.csv"
rawimagesetdir = s3dir+"/images/"    
file_list = os.listdir(rawimagesetdir)
totalnumberofimages = len(file_list)    
    

    
#----------------------        # get starting image         ----------------------#
    

root = tk.Tk()
root.title("String Input")
root.geometry("400x200")  # Set window size to 300x100 pixels
# Define a font with a larger size
font = ("Helvetica", 24)
label = tk.Label(root, text="Input start number:", font=font)
label.pack()
entry = tk.Entry(root, font=font)
entry.pack()
button = tk.Button(root, text="Next", command=save_input, font=font)
button.pack()
root.mainloop()

if user_input == "" :
    # use default
    imagenumber_start =  0

else :
    imagenumber_start = int(user_input)
    #----------------------        # get last image         ----------------------#

root = tk.Tk()
root.title("String Input")
root.geometry("400x200")  # Set window size to 300x100 pixels
# Define a font with a larger size
font = ("Helvetica", 24)
label = tk.Label(root, text="Input end number:", font=font)
label.pack()
entry = tk.Entry(root, font=font)
entry.pack()
button = tk.Button(root, text="Next", command=save_input, font=font)
button.pack()
root.mainloop()

if user_input == "" :
    # use default
    imagenumber_end =  totalnumberofimages

else :
    imagenumber_end = int(user_input)  
    
#----------------------        # get skipping number         ----------------------#
root = tk.Tk()
root.title("String Input")
root.geometry("400x200")  # Set window size to 300x100 pixels
# Define a font with a larger size
font = ("Helvetica", 24)
label = tk.Label(root, text="Input skipping number:", font=font)
label.pack()
entry = tk.Entry(root, font=font)
entry.pack()
button = tk.Button(root, text="Start code", command=save_input, font=font)
button.pack()
root.mainloop()

if user_input == "" :
    # use default
    skippingnumber =  1

else :
    skippingnumber = int(user_input)
    
    #----------------------        # GUI end         ----------------------#

    
imagenumber = 600
# Get a list of files in the directory
file_list = os.listdir(rawimagesetdir)
totalnumberofimages = len(file_list)
# Sort the list of files
sorted_file_list = sorted(file_list)
image_path = rawimagesetdir+sorted_file_list[imagenumber]

    
    
# check if post_object_dection folder exists if not create 

postobjectdetection_path = s3dir + "postobjectdetection/"

# Check if the folder exists
if not os.path.exists(postobjectdetection_path):
    # If it doesn't exist, create it
    os.makedirs(postobjectdetection_path)
    print("postobjectdetection folder created successfully.")
else:
    print("postobjectdetection folder already exists.")
    
detections_path = s3dir + "detection/"

# Check if the folder exists
if not os.path.exists(detections_path):
    # If it doesn't exist, create it
    os.makedirs(detections_path)
    print("detections_path folder created successfully.")
else:
    print("detections_path folder already exists.")    
    
    
    
# add GUI tracker stuff 
import tkinter as tk

def count(i, imagenumber_end, time_sofar):
    expected_total_time = imagenumber_end / i * time_sofar
    ETA = expected_total_time - time_sofar
    label1.config(text="Loops completed = {}/{}".format(int(i), int(imagenumber_end)))
    label2.config(text="ETA = {} (min)".format(int(ETA/60)))
    root.update()

root = tk.Tk()
root.title("Counting GUI")

label1 = tk.Label(root, text="", font=("Helvetica", 24))
label1.pack(pady=10)

label2 = tk.Label(root, text="", font=("Helvetica", 24))
label2.pack(pady=10)
time_start = time.time()



# ---------------------------------------------- main body of code ---------------------------------

keepdata = 0 
imagenumber = imagenumber_start
data = []
pos_data_X = []
pos_data_Y = []
variablelenght = []
Object_tracker = CentroidTracker()
tomswindow = 0.75 
higestIDnumber = 1

scalefactor = 793
scalefactor = 1 
numbers_list = np.arange(0, 255) * 0 
numbers_series = pd.Series(numbers_list)

max_wait_time = 35

df_volume , df_xpos , df_ypos , df_ID, df_meanradi , imagenumber , df_yrange , df_xrange = setupdataframes(keepdata,imagenumber_start)
backupcounter = 0 
loopnumber = 0
while imagenumber < imagenumber_end:
    # manage functions
    os.chdir(here)
    
    print(imagenumber)

    local_time_start = time.time()
    timetype1 = time.time()
    image_path = rawimagesetdir+sorted_file_list[imagenumber]
    imagename = sorted_file_list[imagenumber]
    image = Image.open(image_path) 
    width, height = image.size
    timetype2 = time.time()

    # process the image to cut it up prepare for machine vision then apply AI object detection to cut up images

    imagedraw, image2 = processandimage(image,postobjectdetection_path,detections_path,imagenumber,imagename)
    timetype3 = time.time()

    
    # do the machine vision to put bubble detections back together
    volumedata , X_data , Y_data, meanradi ,radiset1 , radiset2 = machinevisionstuff(imagedraw,scalefactor,tomswindow)
    timetype4 = time.time()

       # pandas nxm arrays/dataframe 
    df_volume , df_xpos, df_ypos , df_ID ,df_meanradi , df_xrange , df_yrange = Writetodataframes(imagenumber,volumedata,X_data,Y_data,df_volume,df_xpos,df_ypos,df_ID,df_meanradi,meanradi,df_xrange , df_yrange,radiset1,radiset2)
    numberofbubbles = len(radiset1)
    # update ID 
    df_ID , higestIDnumber = Create_bubble_IDs(df_xpos,df_ypos,df_ID,df_xrange,df_yrange,imagenumber,numberofbubbles, width, height, higestIDnumber)
    
    columnname = "time = "+str(imagenumber)
    ID_series = df_ID[columnname]
    # Drop NaN values
    ID_valuesofthisstep = ID_series.dropna().tolist()

    draw = ImageDraw.Draw(image2)   
    
    drawingIDinteration = 0 
    while drawingIDinteration < numberofbubbles : 
        
        ID = ID_series[drawingIDinteration] 
        Xpos=X_data[drawingIDinteration]
        Ypos=Y_data[drawingIDinteration]
        
            # write ID on bubbles 
        # Define the text and font
        text = "ID = "+str(ID)
        font = ImageFont.truetype("arial.ttf", 30)

        # Determine the position to put the text
        position = (Xpos- 200, Ypos)  # (x, y)

        # Set the text color
        text_color = (0, 0, 255)  # green

        # Put the text on the image
        draw.text(position, text, fill=text_color, font=font)
        
        drawingIDinteration = drawingIDinteration + 1 
    # draw tom's box 
    TWgap = (1- tomswindow)/2
    width, height = image2.size

    # get ranges for tom's window 
    TW_Xmax =  width * (1- TWgap)
    TW_Xmin =  width * (TWgap)

    TW_Ymax =  height * (1- TWgap)
    TW_Ymin =  height * (TWgap)
    draw.rectangle([TW_Xmin, TW_Ymin, TW_Xmax, TW_Ymax], outline="red", width=5)


    # resave user detector image 
    filename = "Objectdetectionforhumans_"+imagename
    savingname = detections_path + "/" + filename
    image2.save(savingname)

    
    
    
    local_time_end = time.time()
    #try :
        
    try :
        clear_output()
        print("cleared text dump")   
    except :
        print("failed to clear")

    elapsed_time = local_time_end - timetype1
    step1 = timetype2 - timetype1
    step2 = timetype3 - timetype2
    step3 = timetype4 - timetype3
    print("finished loop moving to next")
    
    print(f"Time taken for loop: {elapsed_time} seconds")
    print(f"Number of bubbles: {len(meanradi)} ")
    try: 
        print(f"IDs of bubbles: {ID_valuesofthisstep} ")
        
    except : 
        print("no ID data")
    timetype5 = time.time()
    step5 = timetype5 - timetype4

 
        # prepare for next loop
    imagenumber = imagenumber+skippingnumber
    print(f"now doing: {imagenumber} loop")
    loopnumber = loopnumber + 1
    backupcounter = backupcounter + 1 

        # do GUI stuff

    time_now = time.time()
    time_sofar = time_now - time_start
    count((imagenumber-imagenumber_start)/skippingnumber, (imagenumber_end-imagenumber_start)/skippingnumber, time_sofar)

    

    
    if backupcounter > 10 :

        print("backing up completed so far saving to dataorganisationfolder")
        # Specify the file path where you want to save the CSV file
        csv_file_path = s3dir+'/df_volume.csv'

        # Writing the DataFrame to a CSV file
        df_volume.to_csv(csv_file_path, index=False)

        # Specify the file path where you want to save the CSV file
        csv_file_path = s3dir+'/df_xpos.csv'

        # Writing the DataFrame to a CSV file
        df_xpos.to_csv(csv_file_path, index=False)

        # Specify the file path where you want to save the CSV file
        csv_file_path = s3dir+'/df_ypos.csv'

        # Writing the DataFrame to a CSV file
        df_ypos.to_csv(csv_file_path, index=False)

        # Specify the file path where you want to save the CSV file
        csv_file_path = s3dir+'/df_ID.csv'

        # Writing the DataFrame to a CSV file
        df_ID.to_csv(csv_file_path, index=False)


        # Specify the file path where you want to save the CSV file
        csv_file_path = s3dir+'/df_meanradi.csv'

        # Writing the DataFrame to a CSV file
        df_meanradi.to_csv(csv_file_path, index=False) 
            
        
        backupcounter = 1
        
os.chdir(here)

# finished last loop now saving df to csv
# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_volume.csv'

# Writing the DataFrame to a CSV file
df_volume.to_csv(csv_file_path, index=False)

# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_xpos.csv'

# Writing the DataFrame to a CSV file
df_xpos.to_csv(csv_file_path, index=False)

# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_ypos.csv'

# Writing the DataFrame to a CSV file
df_ypos.to_csv(csv_file_path, index=False)

# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_ID.csv'

# Writing the DataFrame to a CSV file
df_ID.to_csv(csv_file_path, index=False)


# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_meanradi.csv'

# Writing the DataFrame to a CSV file
df_meanradi.to_csv(csv_file_path, index=False) 


# ------------------------------------ main body of code done -------------------------------------


#------------------------------------ process data and make plots and figures --------------------



#------------------------------------ load data sets
scalefactor = 1 
# temp loaction delete later 
#radi 
csv_file_path = s3dir+'/df_meanradi.csv'
df_meanradi = pd.read_csv(csv_file_path)
# positions
csv_file_path = s3dir+'/df_xpos.csv'
df_xpos = pd.read_csv(csv_file_path)
csv_file_path = s3dir+'/df_ypos.csv'
df_ypos = pd.read_csv(csv_file_path)
# ID 
csv_file_path = s3dir+'/df_ID.csv'
df_ID = pd.read_csv(csv_file_path)


# load data sets
print(spreadsheetdir)
try :
    df_cyclingdata = pd.read_csv(spreadsheetdir)
except : 
    try : 
        spreadsheetdir_2 = spreadsheetdir.replace(".csv", ".xlsx")
        df_cyclingdata = pd.read_csv(spreadsheetdir_2)
    except : 
        df_cyclingdata = pd.read_excel(spreadsheetdir_2, sheet_name="record")

print(df_cyclingdata)


# general out put dataframe 
df_overall = pd.DataFrame()

    
label1.config(text="AI stuff completed")
label2.config(text="Making plots {}/{}".format(1, 5))
root.update()    
    
    
# --------------------------- start making plots ----------

        # ---------- get time data 
    
# getting plots
#df_cyclingdata
current = df_cyclingdata['Current(A)']
date = df_cyclingdata['Date']
len(date)
# get datetime data from csv 
currenttimes = pd.Series
imagetimes = pd.Series

currenttimes_list = []
imagetimes_list = []


i= 0 
while i < len(date) : 
    datedata = date[i]
    # Parse the string into a datetime object
    try : 
        datetime_object = datetime.strptime(datedata, '%Y-%m-%d %H:%M:%S')
    except : 
        datetime_object = datetime.strptime(datedata, "%d/%m/%Y %I:%M:%S %p")
    # Get the numeric representation (timestamp) of the datetime object
    timestamp = datetime_object.timestamp()
    currenttimes_list.append(timestamp)
    i=i+1
    #print(timestamp)
# get datetime data from images
file_list = os.listdir(rawimagesetdir)
totalnumberofimages = len(file_list)
# Sort the list of files
sorted_file_list = sorted(file_list)
i=0
while i < len(sorted_file_list) : 

    filename = sorted_file_list[i]
    date_time_str = filename.split('_')[1].split('.')[0]  # Extracts '24-02-15-15-22-45'
    datetime_object = datetime.strptime(date_time_str, '%y-%m-%d-%H-%M-%S')
    timestamp = datetime_object.timestamp()
    imagetimes_list.append(timestamp)
    #print(timestamp)
    #print(datetime_object)
    i=i+1
    
currenttimes = pd.Series(currenttimes_list)
imagetimes = pd.Series(imagetimes_list)


# ---------------------- plot area covered vs time 


# plotting area vs time 
time_list = [] 
total_area_list = []


imagenumber_start = 0
timeend = len(imagetimes)
#timeend = 240
step = 1 
print(timeend)

step = 10
imagenumber = imagenumber_start
counter = 0
numberofsteps = 600
minarea = 0.01

point_size = 10

while imagenumber < timeend:
#while counter < numberofsteps:
    # Plotting the scatter plot
    columnname = "time = "+str(imagenumber)
    #columnname = df_volume.columns[counter]
    try : 
        meanradi_series = df_meanradi[columnname]
        gotcolumn = 1 
    except : 
        gotcolumn = 0 
        #print(columnname + " doesn't exist")
        counter = counter - 1
    if gotcolumn == 1 :
        area_series = meanradi_series* meanradi_series * math.pi *2 # *2 as volume stored assuming hemispheres 
        datapoints = area_series.count()

        i= 0 
        Area_instant_total = 0
        while i < datapoints : 

            area_single = area_series[i]
            if area_single > minarea :
                Area_instant_total = area_single + Area_instant_total

            i=i+1
        #time_list[counter] = imagenumber 
        #total_area_list[counter] = Area_instant_total
        time_list.append(imagenumber)
        total_area_list.append(Area_instant_total)
    imagenumber = imagenumber + step
    counter = counter + step
    #print(imagenumber)

    # convert list to series to columns in a dataframe 

total_area_series = pd.Series(total_area_list)

fraction_area_series = total_area_series / (3840 * 2160) * 100

time_series = pd.Series(imagetimes)
# Create a DataFrame
df_coverage = pd.DataFrame()
# Add the Pandas Series as a column to the DataFrame
df_coverage['total_area'] = total_area_series
df_coverage['time'] = imagetimes 

start_time = currenttimes[0]
timelisthourin = imagetimes[time_list] - start_time
timelisthourin = [x / 60 /60 for x in timelisthourin]

    # Show the plot
ax = plt.scatter(x=timelisthourin, y=fraction_area_series, color='blue', marker='o',s=point_size)
x_min = timelisthourin[0]
x_max = timelisthourin[-1]
plt.xlim(x_min, x_max)  # Set the x-axis range from 0 to 'x_max'
plt.title("Area_covered_vs_Time")
plt.xlabel('Time (hours)')
plt.ylabel('Percentage of Area covered (%)')
plt.savefig(s3dir+"Area_covered_vs_Time.png")
plt.show()

     # save the data
# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_coverage.csv'

# Writing the DataFrame to a CSV file
df_coverage.to_csv(csv_file_path, index=False)



label1.config(text="AI stuff completed")
label2.config(text="Making plots {}/{}".format(3, 5))
root.update()


# plot current vs time

start_time = currenttimes[0]
currenttimes_zeroed = currenttimes - start_time
currenttimes_zeroed = [x / 60 /60 for x in currenttimes_zeroed]

# conversion factor mm radius of o ring to area of electrode
oringradius = 7.5/10 # cm 
ortingarea = oringradius*oringradius*3.14
print(ortingarea)
    # Show the plot
point_size = 1
#ax = plt.scatter(x=currenttimes_zeroed, y=current/ortingarea, color='blue', marker='o',s=point_size)
ax = plt.plot(currenttimes_zeroed, current/ortingarea, color='blue', marker='o', markersize=point_size, linestyle='-')

plt.xlim(x_min, x_max)  # Set the x-axis range from 0 to 'x_max'

plt.title("Current vs Time")
plt.xlabel('Time (hours)')
plt.ylabel('Current (mA/cm2)')
plt.savefig(s3dir+"Current vs Time.png")
plt.show()


label1.config(text="AI stuff completed")
label2.config(text="Making plots {}/{}".format(4, 5))
root.update()

# get data from csv 
file_path = s3dir+'/df_meanradi.csv'
df_meanradi = pd.read_csv(file_path)

file_path = s3dir+'/df_xpos.csv'
df_xpos = pd.read_csv(file_path)

file_path = s3dir+'/df_ypos.csv'
df_ypos = pd.read_csv(file_path)

file_path = s3dir+'/df_ID.csv'
df_ID = pd.read_csv(file_path)



# make folder for heatmap transition
heatsink_path = s3dir + "heatsink/" 
# Check if the folder exists
if not os.path.exists(heatsink_path):
    # If it doesn't exist, create it
    os.makedirs(heatsink_path)
    print("heatsink_path folder created successfully.")
else:
    print("heatsink_path folder already exists.")


timeend = imagenumber_end
step = 1 


height, width = 2160, 3860
heatmap_array = np.zeros((height, width))

def draw_circle(array, x, y, radius):
    ny, nx = array.shape
    Y, X = np.ogrid[:ny, :nx]
    mask = (X - x)**2 + (Y - y)**2 <= radius**2
    array[mask] += 1
    
    
    
    
step = 1
imagenumber = imagenumber_start
counter = 0
minarea = 0

point_size = 10
duration = 0
while imagenumber < timeend:
    columnname = "time = "+str(imagenumber)

    try : 
        meanradi_series = df_meanradi[columnname]
        xpos_series = df_xpos[columnname]
        ypos_series = df_ypos[columnname]
        gotcolumn = 1 
        duration = duration + 1 
        #print(columnname)
    except : 
        gotcolumn = 0 
        #print(columnname + " doesn't exist")
        counter = counter - 1
    
    
    if gotcolumn == 1 :
        datapoints = meanradi_series.count()
        
        i= 0 
        
        while i < datapoints : 

            meanradi_single = meanradi_series[i]
            xpos_single = xpos_series[i]
            ypos_single = ypos_series[i]
            area_single = meanradi_single*2*3.14
            if area_single > minarea :
                # add to map 

                
                
                draw_circle(heatmap_array, xpos_single, ypos_single, meanradi_single*scalefactor)
                #draw_circle(image_array, 1000, 1000, 100)
                

            i=i+1
        #print(imagenumber)
            
    # save copy of heatmap
        
            # standardise 

        heatmap_adjusted_array = heatmap_array
        heatmap_adjusted_array = heatmap_adjusted_array / (duration+1)#np.max(heatmap_array)

            # convert np to plt and save 
        plt.imshow(heatmap_adjusted_array, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        plt.title("Fraction of time each pixels is covered")
        cbar = plt.colorbar(label="fraction of time covered")
        imagenumberstring =  "{:05d}".format(imagenumber)
        heatsinkname = "heatsink_"+imagenumberstring+".png"
        print(heatsinkname)
        plt.savefig(heatsink_path+heatsinkname)
    #plt.show()    
    # Reset the plot
    plt.clf()  # or plt.close()
    
    
    imagenumber = imagenumber + step
    #print(imagenumber)
    counter = counter + step
    #print(imagenumber)
    
    # convert list to series to columns in a dataframe 

heatmap_array = np.flipud(heatmap_array)
heatmap_adjusted_array = heatmap_array / duration#np.max(heatmap_array)
column_titles = df_meanradi.columns.tolist()

plt.imshow(heatmap_adjusted_array, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
plt.title("Fraction of time each pixels is covered")
cbar = plt.colorbar(label="fraction of time covered")
plt.savefig(s3dir+"Heatmap.png")
plt.show()

label1.config(text="Static plots finished")
label2.config(text="Making Video")
root.update()


# get time data 
column_titles = df_meanradi.columns.tolist()

# Regular expression pattern to extract the number
pattern = r'\d+'

# Extract numbers from each string using list comprehension
time_data = [int(entry.split('=')[1].strip()) for entry in column_titles[1:]]

time_data_series = pd.Series(time_data)


series_length = len(time_data_series)
Highest_bubble_ID = df_ID.max().max() 
Arrayofradi = np.full((series_length ,int(Highest_bubble_ID)), np.nan)

loopnumber = 1
while loopnumber < len(column_titles):
    columnname = column_titles[loopnumber]
    meanradi_series = df_meanradi[columnname]
    ID_series = df_ID[columnname]
 
    row = 0 
    while row < len(ID_series) : 
        bubble_ID_number = ID_series[row]
        if not np.isnan(bubble_ID_number):
            #print("bubble_ID_number is not NaN.")
            inputmeanradi_value = meanradi_series[row]
            Arrayofradi[loopnumber-1][int(bubble_ID_number)-1] = inputmeanradi_value
            row = row + 1
        else:
            #print("bubble_ID_number is NaN.")
            row = len(ID_series) * 100 # end loop
            
        
    
    
    loopnumber = loopnumber + 1 
df_trackedradi = pd.DataFrame(Arrayofradi)
new_columns = []
i= 0
while i < Highest_bubble_ID:
    
    new_column =  str(i)+"_bubble"
    new_columns.append(new_column)
    i=i+1 
    
    
# Change column titles
df_trackedradi.columns = new_columns






bubblenumber = 1
while bubblenumber < Highest_bubble_ID:
#while bubblenumber < 10:     
    columnname = str(bubblenumber)+"_bubble"
    IDed_radi_series = df_trackedradi[columnname]

    volume_estimate = 4/3 * 3.14 * (IDed_radi_series*IDed_radi_series*IDed_radi_series)


    ax = plt.scatter(x=time_data_series/60/4, y=volume_estimate,s=1)
    try :
        clear_output()
        print("cleared text dump")   
    except :
        print("failed to clear")
        
    print(bubblenumber)
    
    
    
    bubblenumber = bubblenumber + 1 
plt.title("Each Bubble's Volume vs Time")
plt.xlabel('Time (hours)')
plt.ylabel('Volume (length of a pixel cubed idk)')
plt.savefig(s3dir+"each_bubbles's_volume.png")
plt.show()




# plotting total volume vs time 
time_list = [] 
total_volume_list = []


imagenumber_start = 0
timeend = len(imagetimes)
#timeend = 240
step = 1 
print(timeend)

step = 1
imagenumber = imagenumber_start
counter = 0
numberofsteps = 600
minarea = 0.01

point_size = 10

while imagenumber < timeend:
#while counter < numberofsteps:
    # Plotting the scatter plot
    columnname = "time = "+str(imagenumber)
    #columnname = df_volume.columns[counter]
    try : 
        meanradi_series = df_meanradi[columnname]
        gotcolumn = 1 
    except : 
        gotcolumn = 0 
        #print(columnname + " doesn't exist")
        counter = counter - 1
    if gotcolumn == 1 :
        volume_series = (meanradi_series* meanradi_series* meanradi_series) * 3.14 *4/3 # add a fudge factor for capped sphere 
        datapoints = volume_series.count()

        i= 0 
        volume_instant_total = 0
        while i < datapoints : 

            volume_single = volume_series[i]
            if area_single > minarea :
                volume_instant_total = volume_single + volume_instant_total

            i=i+1
        #time_list[counter] = imagenumber 
        #total_area_list[counter] = Area_instant_total
        time_list.append(imagenumber)
        total_volume_list.append(volume_instant_total)
    imagenumber = imagenumber + step
    counter = counter + step
    #print(imagenumber)

    # convert list to series to columns in a dataframe 

total_volume_series = pd.Series(total_volume_list)
volume_series = total_volume_series 

time_series = pd.Series(imagetimes)
# Create a DataFrame
df_coverage = pd.DataFrame()
# Add the Pandas Series as a column to the DataFrame
#df_volume['total_volume'] = total_volume_series
#df_coverage['time'] = imagetimes 

start_time = currenttimes[0]
timelisthourin = imagetimes[time_list] - start_time
timelisthourin = [x / 60 /60 for x in timelisthourin]

    # Show the plot
ax = plt.scatter(x=timelisthourin, y=total_volume_series, color='blue', marker='o',s=point_size)

x_min = timelisthourin[0]
x_max = timelisthourin[-1]
plt.xlim(x_min, x_max)  # Set the x-axis range from 0 to 'x_max'
plt.title("Total volume_vs_Time")
plt.xlabel('Time (hours)')
plt.ylabel('volume (idk)')
plt.savefig(s3dir+"Total_volume_vs_Time.png")
plt.show()

     # save the data
# Specify the file path where you want to save the CSV file
csv_file_path = s3dir+'/df_volume.csv'

# Writing the DataFrame to a CSV file
df_coverage.to_csv(csv_file_path, index=False)



# get growth rate 
samplingrate= 30
bubblenumber = 1
while bubblenumber < Highest_bubble_ID:
    
    columnname = str(bubblenumber)+"_bubble"
    IDed_radi_series = df_trackedradi[columnname]

    volume_estimate = 4/3 * 3.14 * (IDed_radi_series*IDed_radi_series*IDed_radi_series)
    
    rate_of_change = volume_estimate.diff(periods=60)  # Difference between consecutive values
    
    # adjust value for sampling rate 
    timeofeachstep = (time_data_series[1]/60/4-time_data_series[0]/60/4) # time step in hours 
    rate_of_change = rate_of_change / timeofeachstep 
    
    
    if bubblenumber == 1 :
        rate_of_change = rate_of_change.fillna(0)
        totalchange = rate_of_change 
    if bubblenumber > 1 :
        rate_of_change = rate_of_change.fillna(0)
        totalchange = rate_of_change + totalchange
    
    #ax = plt.scatter(x=time_data_series/60/4, y=rate_of_change,s=1)

    bubblenumber = bubblenumber + 1 

ax = plt.scatter(x=time_data_series/60/4, y=totalchange,s=point_size/2)
plt.title("growth rate vs Time")
plt.xlabel('Time (hours)')
plt.ylabel('Growth rate (length of a pixel cubed idk / hour) ')
plt.savefig(s3dir+"Each_bubbles's_growthrate.png")
plt.show()


import matplotlib.pyplot as plt

# Sample data
x=time_data_series/60/4
y1 = totalchange # First set of data
y2 = total_volume_series  # Second set of data

# Creating the first plot
fig, ax1 = plt.subplots()

# Plotting the first set of data as scatter plot
ax1.scatter(x, y1, color='black', label='Scatter 1',s=1)  
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Volume change (ml/h)', color='black')
ax1.tick_params('y', colors='black')

# Creating the second plot with its own y-axis
ax2 = ax1.twinx()
ax2.scatter(x, y2, color='blue', label='Scatter 2',s=1)
ax2.set_ylabel('Volume (ml)', color='blue')
ax2.tick_params('y', colors='blue')

# Adding title
plt.title('Volume and Change in volume over time')

# Displaying the plot
plt.savefig(s3dir+"Volume_totalandchange.png")
plt.show()


# get growth rate 
samplingrate= 30
bubblenumber = 1
while bubblenumber < Highest_bubble_ID:
     
    columnname = str(bubblenumber)+"_bubble"
    IDed_radi_series = df_trackedradi[columnname]

    volume_estimate = 4/3 * 3.14 * (IDed_radi_series*IDed_radi_series*IDed_radi_series)
    
    rate_of_change = volume_estimate.diff(periods=samplingrate)  # Difference between consecutive values
    timeofeachstep = (time_data_series[1]/60/4-time_data_series[0]/60/4) 
    rate_of_change = rate_of_change / timeofeachstep / samplingrate
    
    ax = plt.scatter(x=time_data_series/60/4, y=rate_of_change,s=1)

    bubblenumber = bubblenumber + 1 


plt.title("growth rate vs Time")
plt.xlabel('Time (hours)')
plt.ylabel('Growth rate (length of a pixel cubed idk / hour) ')
plt.savefig(s3dir+"Each_bubbles's_growthrate.png")
plt.show()



# ------------------------- making video 


# make folder for heatmap transition
vidframe_dump_path = s3dir + "vidframe_dump/" 
# Check if the folder exists
if not os.path.exists(vidframe_dump_path):
    # If it doesn't exist, create it
    os.makedirs(vidframe_dump_path)
    print("vidframe_dump_path folder created successfully.")
else:
    print("vidframe_dump_path folder already exists.")



# get images list of images in folders 

heatsink_list = os.listdir(heatsink_path)
detection_human_path = s3dir + "detection/" 
detection_human_list = os.listdir(detection_human_path)

# get static images
electrodataplot_PIL = Image.open(s3dir + "Current vs Time.png")
coverageplot_PIL = Image.open(s3dir + "Area_covered_vs_Time.png")
Area_indivdiaulplot_PIL = Image.open(s3dir + "Each_bubbles_Area.png")
#Volume_PIL = Image.open(s3dir + "Volume_totalandchange.png") 
Volume_PIL = Image.open(s3dir + "Total_volume_vs_Time.png") 



# loop stuff

imagenumber = 0 
while imagenumber < len(heatsink_list) :
    #in loop

    # get changing images 
    heatmap_PIL = Image.open(heatsink_path+heatsink_list[imagenumber])
    detection_PIL =  Image.open(detection_human_path+detection_human_list[imagenumber])

    # put 4 images together 

    # Determine the size of the final image
    width, height = detection_PIL.size  # Assuming all images are of the same size
    oldsize = (width, height)
    # Create a new image with a size large enough to accommodate all four images
    new_width = width * 2
    new_height = height * 2
    combined_image = Image.new("RGB", (int(new_width*1.1), int(new_height*1.1)), color="white")
    
    # grab color bar make smaller 
        # crop to be just image
    box = [ 80 , 131 ,478 , 354]
    heatmap_PIL_justimage = heatmap_PIL.crop(box)
    heatmap_PIL_justbar = heatmap_PIL.crop(box)
    # rescale images # 4:3 for plots 16∶9 for collected images

    new_size = (int(height*4/3), height)  # Replace width and height with your desired dimensions
    Volume_PIL = Volume_PIL.resize(new_size)
    coverageplot_PIL = coverageplot_PIL.resize(new_size)
    electrodataplot_PIL = electrodataplot_PIL.resize(new_size)

    # down scale dectionimage so heatmap betterfits
    factor_down = 1
    down_size = (int(width*factor_down), int(height*factor_down))
    detection_PIL = detection_PIL.resize(down_size)
    
    # do rescale for heatmap
    heatmap_map_PIL = heatmap_PIL_justimage.resize(down_size)
        
        
    
    # Paste each image into the combined image
    combined_image.paste(heatmap_map_PIL, (0 , height))
    #combined_image.paste(detection_PIL, (0+int((1-factor_down)/2*width), int((1-factor_down)*height)))
    combined_image.paste(detection_PIL, (0, 0))
    combined_image.paste(Volume_PIL, (width+10+200, 0))
    combined_image.paste(coverageplot_PIL, (width+10+200, height))

    
    # draw timeline on 
    
    xzero = 4211 + 200
    xend  = 6442 + 200
    
    line_scalefactor = (xend-xzero)
    
    xnow = line_scalefactor * (imagenumber) / (len(heatsink_list)-1) + xzero
    
    
    # Draw a vertical line
    draw = ImageDraw.Draw(combined_image)
    x_coordinate = 300  # Specify the x-coordinate where you want the line
    line_color = (0, 0, 0)  # Specify the color of the line (black in this case)
    line_start = (xnow, 0)  # Start point of the line
    line_end = (xnow, new_height)  # End point of the line
    draw.line([line_start, line_end], fill=line_color, width=10)  # Adjust width as needed
    
    #display(combined_image)

    imagenumberstring =  "{:05d}".format(imagenumber)
    vidframename = "vidframe"+imagenumberstring+".png"
    
    combined_image = combined_image.resize(oldsize)
    
    combined_image.save(vidframe_dump_path+vidframename)
    
    #display(combined_image)
    imagenumber = imagenumber+1
    try :
        clear_output()
        print("cleared text dump")   
    except :
        print("failed to clear")
    print(imagenumber/ len(heatsink_list))
#display(combined_image)



# make mp4 

# Path to the folder containing the images
folder_path = vidframe_dump_path

# Get the list of image filenames in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]

# Sort the filenames to ensure the images are in order
image_files.sort()

# Specify the output video file name
output_video = "output.mp4"

# Get the dimensions of the first image to set the video size
first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
height, width, _ = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, 1, (width, height))
i=0 
# Iterate through the image files and add them to the video
while i < imagenumber  : 
#for image_file in image_files:
    image_file = image_files[i]
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)
    out.write(frame)
    try :
        clear_output()
        print("cleared text dump")   
    except :
        print("failed to clear")
    print(i/imagenumber)
    i=i+1

# Release the VideoWriter and close all windows
out.release()
cv2.destroyAllWindows()


# Move the video file to the s3dir directory
output_video_path = os.path.join(s3dir, output_video)
if os.path.exists(output_video_path):
    os.remove(output_video_path)  # Remove existing file
os.rename(output_video, output_video_path)


print("Video created successfully!")



# make universal csv 
oringarea = 1 
# col 1 = time
# col 2 = number of bubbles 
# col 3 = Volume 
# col 4 = change of voluem 
# col 5 = Area covered 
# col 6 = Area of Bubble electrolye interface 
# col 7 = Electrodatatime
# col 8 = Current 
# col 9 = Voltage

# Creating an empty DataFrame
General_total_df = []

General_total_df = pd.DataFrame(columns=['Time_images', 'Number of bubbles', 'Volume','Change of Volume' , 'Area covered fractional','Area of Bubble electrolye interface','Time_electro_data'])

General_total_df['Time_images'] = timelisthourin

General_total_df['Volume'] = total_volume_series

General_total_df['Change of Volume'] = totalchange

General_total_df['Area covered fractional'] = fraction_area_series

#General_total_df['Area of Bubble electrolye interface'] = interface_area_series



#General_total_df['Voltage'] = Voltage

print(General_total_df)









label1.config(text="Everything finished")
label2.config(text="Making plots {}/{}".format(5, 5))
root.update()



