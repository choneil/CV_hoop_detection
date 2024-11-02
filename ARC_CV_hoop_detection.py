import cv2 as cv
import numpy as np
import math
from sortlines import sort_lines

vid = cv.VideoCapture('IMG_3936.mov')

#Dimensions of video frame
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

#Variables to resize video frame with the correct ratio
ratio = width/height
new_h = 1080
new_w = int(new_h*ratio) 

while True:

    
    ret, frame = vid.read()
    
    #Resize frame while maintaining ratio
    small_frame = cv.resize(frame,(1080,920))
    #Convert color to HSV 
    hsv = cv.cvtColor(small_frame, cv.COLOR_BGR2HSV)

    #Convert to grayscale
    gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)

    #Lower span of red in HSV coordinates
    lower_red1 = np.array([0,100,150])
    upper_red1 = np.array([15,255,255])

    #Mask the Lower span of red in the frame. All masked pixels will be white while others are black
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)

    #Upper span of red in HSV coordinates
    lower_red2 = np.array([170,100,150])
    upper_red2 = np.array([180,255,255])

    #Mask the upper span of red in the frame. All masked pixels will be white while others are black
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    #Combine the lower and upper red masks
    mask_all = mask1 + mask2

    #Morphology using the combined mask as the source parameter, and (7,7) as the kernel
    #Explanation of Morphology commands(OPEN, DILATE, etc.) in [1]
    mask_all = cv.morphologyEx(mask_all, cv.MORPH_OPEN, np.ones((7,7),np.uint8))
    mask_all = cv.morphologyEx(mask_all, cv.MORPH_DILATE, np.ones((7,7),np.uint8))
    
    #More filters to try to hone in on straight lines
    thresh = cv.threshold(mask_all, 130, 255, cv.THRESH_BINARY)[1]
    blur = cv.GaussianBlur(thresh,(9,9),10,None,1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    dilate = cv.morphologyEx(blur, cv.MORPH_DILATE, kernel)
    
    # get absolute difference between dilate and the blurred threshold
    # got better results for my lines doing this rather than just the threshold
    diff = cv.absdiff(dilate, blur)

    #invert diff
    edges = 255 - diff

    #canny edges of the diff 
    can = cv.Canny(diff,100,255,None,3,False)

    #convert the inverted diff to 'color'
    can_c = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    
    #leftover from, useless
    kernel = np.ones((5,5), np.uint8)
    
    dst = cv.Canny(blur, 10, 200, None, 3)
    #eroded_image = cv.erode(blur, kernel,None,(1,1),5)
    
    cdst = cv.cvtColor(can, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(can, 1, np.pi / 180, 200, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

#using predicted hough lines here because it seemed like the results were more useful for what I was trying to do
#it was giving more results we could at least filter down to something close to accurate

    
    linesP = cv.HoughLinesP(can, 1, np.pi / 180, 10, None, 100, 20)
    #create an empty array for horizontal both horizontal and vertical lines
    l_h=[]
    l_v=[]

    
    if linesP is not None:
        for i in range(0, len(linesP)):


            l = linesP[i][0]
            cv.line(cdstP,(l[0],l[1]),(l[2],l[3]),(0,0,255),3,cv.FILLED)
            dxdy =abs((l[2]-l[0])/(l[3]-l[1]+.001))
            
            #2d array new_l with dxdy defined by x2-x1/y2-y1. .001 added to denominator to prevent division by zero
            #will clean up some of the slop like that .001 as we go
            new_l=[[dxdy],[l[0],l[1],l[2],l[3]]]
            print(new_l)
           
            #append line into appropriate array
            if dxdy>1:
                l_h.append(new_l)
            else:
                l_v.append(new_l)
    
    
    print(l_h)
    for i in range(0,len(l_h)-1):         
        if(l_h[i][0][0]>l_h[i+1][0][0]):
            
            print(l_h[i])
            l2=l_h[i+1]
            l1=l_h[i]
            l_h[i]=l2
            l_h[i+1]=l1
        h = l_h[i]
        #print('h',h)
        cv.line(cdstP, (int(h[1][0]), int(h[1][1])), (int(h[1][2]), int(h[1][3])), (0,255,0), 3, cv.FILLED)
        #cv.circle(cdstP,(l[0],l[1]), 1,(255,0,0))
        #cv.putText(cdstP,(str(l) + ': ' + str(l[0]) + ',' + str(l[1])), (l[0]+5,l[1]-5),0,.25,(255,0,0))
        #cv.circle(cdstP,(l[2],l[3]), 1, (255,0,0))  
        #cv.putText(cdstP,(str(i) + ': ' + str(l[2]) + ',' + str(l[3])), (l[2]+5,l[3]-5),0,.25,(255,0,0))        
    sort_lines(l_v)
    for i in range(0,len(l_v)):
        v=l_v[i]
        cv.line(cdstP, (int(v[1][0]), int(v[1][1])), (int(v[1][2]), int(v[1][3])), (255,0,0), 3, cv.FILLED)
   
    res = cv.bitwise_and(small_frame,small_frame,mask=mask_all)
    while True:

        cv.imshow('cdstP',cdstP)   
    #numpy_horizontal_concat1 = np.concatenate((cdst,cdstP), axis=1)
    #numpy_horizontal_concat2 = np.concatenate((small_frame,can_c),axis=1)
    #numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2),axis=0)
    #cv.imshow('Hough Lines, Hough Predictions, Original, Filtered', numpy_vertical_concat)
    #cv.imshow('predict',cdstP)
    #cv.imshow('hough',cdst)
    #cv.imshow('mask', edges)
    #cv.imshow('canny',can)
    #cv.imshow('blur',blur)

        if cv.waitKey(1) == ord('q'):
            break
   


    
    
    
        

vid.release()

cv.destroyAllWindows