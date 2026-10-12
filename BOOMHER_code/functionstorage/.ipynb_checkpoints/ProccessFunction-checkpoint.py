import os 
here = os.getcwd()
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
import cv2
import numpy as np
import contextlib
from IPython.utils import io


def processandimage(image,postobjectdetection_path,detections_path,imagenumber,imagename):
    detection_threshold = 0.5
    width, height = image.size
    
    bubblenumber = 1
    model = "retrained.pt"
    model = YOLO(model)
    image2 = image.copy()
    Completelydifferentimage = image
    # Create a new black image with the same size
    black_image = Image.new("RGB", (width, height), color="black")
    
    stepsize = 640
    framesize = 1280

    stepsx = (width -framesize) / stepsize 
    stepsy = (height-framesize) / stepsize 
    x=0 
    y=0
    it = 1
    while (x < stepsx +1) :
        y=0
        while (y<stepsy + 1) : 
            xranges = (0+stepsize*x , framesize + stepsize*x)  
            yranges = (0+stepsize*y , framesize + stepsize*y)

            crop_box = (xranges[0], yranges[0], xranges[1], yranges[1])
 
            cropped_image = image.crop(crop_box)

            # do AI stuff

            color = (255)  # Red color in BGR format (Blue, Green, Red)
            thickness = 5

            #get result
            results = model(cropped_image, verbose=False)
            #results = model(cropped_image, verbose=True)
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
    
                    xshift = xranges[0] 
                    yshift = yranges[0] 

                    x1 = int(x1) 
                    x2 = int(x2) 
                    y1 = int(y1) 
                    y2 = int(y2) 

                    widthlabel = x2 - x1
                    heightlabel = y2 - y1

                    centerX = (x2 + x1)/2 + xshift
                    centerY = (y2 + y1)/2 + yshift

                    X1 = x1 + xshift
                    X2 = x2 + xshift
                    Y1 = y1 + yshift
                    Y2 = y2 + yshift
                    
                    
                    
                    expansion = 0.5 
            
                    # cropping 
                    left = X1 - widthlabel * expansion
                    upper = Y1 - heightlabel * expansion
                    right = X2 + widthlabel * expansion
                    lower = Y2 + heightlabel * expansion

                    # Crop the image
                    cutting_image = Completelydifferentimage.crop((left, upper, right, lower))

                    cropname = str(bubblenumber)+"_" + str(centerX)+"_" + str(centerY)+".png"
                    bubblenumber = bubblenumber + 1
                    savingname = here+"\\cropdump\\"+cropname
                    #cutting_image.save(savingname)
                    
                    
                    
                    class_id = int(class_id)
                    if score > detection_threshold:
                        detections.append([x1, y1, x2, y2,])
                    draw_display = ImageDraw.Draw(image2)
                    draw_display.rectangle([X1, Y1, X2, Y2], outline=(0, 255, 0), width=2)
                    draw_display.line([centerX, Y1, centerX, Y2], fill='white', width=5)
                    draw_display.line([X1, centerY, X2, centerY], fill='white', width=5)
                    draw_blank = ImageDraw.Draw(black_image)
                    draw_blank.line([centerX, Y1, centerX, Y2], fill='white', width=2)
                    draw_blank.line([X1, centerY, X2, centerY], fill='white', width=2)
                    

                    
                    
                    
                    
            it = it + 1 
            y=y+1

        x=x+1
    # add a final image of the orginal uncropped 
    results = model(image, verbose=False)
    for result in results:
        detections = []

        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r

            xshift = xranges[0] 
            yshift = yranges[0] 

            #print(x,y)
            #print(xshift,yshift)

            x1 = int(x1) 
            x2 = int(x2) 
            y1 = int(y1) 
            y2 = int(y2) 

            widthlabel = x2 - x1
            heightlabel = y2 - y1

            centerX = (x2 + x1)/2 
            centerY = (y2 + y1)/2 

            X1 = x1 
            X2 = x2 
            Y1 = y1 
            Y2 = y2 

            expansion = 0.5 
            # cropping 
            left = X1 - widthlabel * expansion
            upper = Y1 - heightlabel * expansion
            right = X2 + widthlabel * expansion
            lower = Y2 + heightlabel * expansion

            # Crop the image
            cutting_image = Completelydifferentimage.crop((left, upper, right, lower))
            
            cropname = str(0)+str(bubblenumber)+"_" + str(centerX)+"_" + str(centerY)+".png"
            bubblenumber = bubblenumber + 1
            savingname = here+"\\cropdump\\"+cropname
            #cutting_image.save(savingname)
            
            
            
            
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2,])
            draw_display = ImageDraw.Draw(image2)
            draw_display.rectangle([X1, Y1, X2, Y2], outline=(0, 255, 0), width=2)
            draw_display.line([centerX, Y1, centerX, Y2], fill='white', width=5)
            draw_display.line([X1, centerY, X2, centerY], fill='white', width=5)
            draw_blank = ImageDraw.Draw(black_image)
            draw_blank.line([centerX, Y1, centerX, Y2], fill='white', width=5)
            draw_blank.line([X1, centerY, X2, centerY], fill='white', width=5)
            
            
            

            
            
    totalnumber = it - 1
    filename = "Objectdetectionforhumans_"+imagename
    savingname = detections_path + "/" + filename
    image2.save(savingname) # saves object detection overlayed onto image for people to have pretty things and to see where it fails 
    
    filename = "Machine_part_"+imagename
    savingname = postobjectdetection_path + "/" + filename
    #black_image.save(savingname) # this saves what the machine sees for next part setup to not save by defult but code exists if needed in future 
    
    
    #print("ending proccessing")
    return black_image , image2