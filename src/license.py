# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os 

path = 'C:/Users/gandh/Desktop/AI_Workshop/test_images/'
for subdir, dirs, files in os.walk(path):
    for file in files:
        re_path = os.path.join(subdir,file)
        og_image = cv2.imread(re_path,1)
        image = cv2.imread(re_path,0)  

        #Resizing the Image
        og_image=cv2.resize(og_image,(500,400))
        image = cv2.resize(image,(500,400))
        #Blurr
        blur= cv2.bilateralFilter(image,11,25,25)
        #cv2.imshow("Blur image", blur)

        #Histogram Equalization
        contrast = cv2.equalizeHist(blur)
        #cv2.imshow("Contrast Image",contrast)

        #Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
        morphed = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, kernel)

        #subtraction
        subtract = cv2.subtract(morphed,contrast)
        #cv2.imshow("Subtracted Image",subtract)

        #Threshold 
        retval, threshold = cv2.threshold(morphed, 200, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Threshold Image",threshold)

        edged = cv2.Canny(threshold, 160, 200,L2gradient = True)
        #cv2.imshow("Canny edged Image", edged)

        #erosion = cv2.erode(gray,kernel,iterations = 1)
        #cv2.imshow("Eroded Image",erosion)

        dilate = cv2.dilate(edged,kernel,iterations = 1)
        cv2.imshow("Dilated Image",dilate)

        (new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
        NumberPlateCnt = None #we currently have no Number plate contour

        count = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05*peri, True)
            if len(approx) == 4:  # Select the contour with 4 corners
                NumberPlateCnt = approx #This is our approx Number Plate Contour
                break

        cv2.drawContours(og_image, [NumberPlateCnt], -1, (0,255,0), 3)
        cv2.imshow("Final Image With Number Plate Detected",og_image)


        #cv2.imshow("blah blah",edged)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()