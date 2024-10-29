import cv2 as cv
import numpy as np
import math

vid = cv.VideoCapture('IMG_3936.mov')

#Dimensions of video frame
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

#Variables to resize video frame with the correct ratio
ratio = width/height
new_h = 540
new_w = int(new_h*ratio) 

while True:

    
    ret, frame = vid.read()
    
    #Resize frame while maintaining ratio
    small_frame = cv.resize(frame,(new_w,new_h))

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
    
    # get absolute difference between dilate and thresh
    diff = cv.absdiff(dilate, blur)

    #more filtering
    edges = 255 - diff 
    can = cv.Canny(diff,100,255,None,3,False)
    can_c = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
   
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

    
    linesP = cv.HoughLinesP(can, 1, np.pi / 180, 10, None, 100, 20)
    print(linesP)
    #segmented=segment_by_angle_kmeans(linesP)
    #print(linesP)
    #print(segmented)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,150), 3, cv.FILLED)
            cv.circle(cdstP,(l[0],l[1]), 1,(255,0,0))
            cv.putText(cdstP,(str(l) + ': ' + str(l[0]) + ',' + str(l[1])), (l[0]+5,l[1]-5),0,.25,(255,0,0))
            cv.circle(cdstP,(l[2],l[3]), 1, (255,0,0))  
            cv.putText(cdstP,(str(i) + ': ' + str(l[2]) + ',' + str(l[3])), (l[2]+5,l[3]-5),0,.25,(255,0,0))
    
    
    
   
    res = cv.bitwise_and(small_frame,small_frame,mask=mask_all)
    #background = cv.bitwise_and(gray,gray,mask=inv)

    #background = np.stack((background,)*3,axis=-1)

    #vid_ca = cv.add(res,background)
    
    #canny = cv.Canny(mask_all, 10, 70)
    #ret, can = cv.threshold(canny, 70, 255, cv.THRESH_BINARY)
    #can = np.stack((can,)*3,axis=-1)
    #new = cv.add(can, background)
   
    
    
        
        
        
    numpy_horizontal_concat1 = np.concatenate((cdst,cdstP), axis=1)
    numpy_horizontal_concat2 = np.concatenate((small_frame,can_c),axis=1)
    numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2),axis=0)
    cv.imshow('Hough Lines, Hough Predictions, Original, Filtered', numpy_vertical_concat)
    #cv.imshow('predict',cdstP)
    #cv.imshow('hough',cdst)
    #cv.imshow('mask', edges)
    #cv.imshow('canny',can)
    #cv.imshow('blur',blur)

    if cv.waitKey(1) == ord('q'):
        break
#cv.imshow('mak',canny)
    #cv.imshow('bonk',inv)
    #cv.imshow('clonk',vid_ca)
    #cv.imshow('chonk', background)
    #cv.imshow('sponk', mask_all)


    
    
    
        

vid.release()

cv.destroyAllWindows