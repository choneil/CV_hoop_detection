import cv2 as cv
import numpy as np
import math
from maxmin  import maxmin
from points import line_intersection
vid = cv.VideoCapture('IMG_3936.mov')

#Dimensions of video frame
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

#Variables to resize video frame with the correct ratio
ratio = width/height
new_h = 720
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
    
    # get absolute difference between dilate and the blurred threshold
    # got better results for my lines doing this rather than just the threshold
    diff = cv.absdiff(dilate, blur)

    #invert diff
    edges = 255 - diff

    #canny edges of the diff 
    can = cv.Canny(diff,100,255,None,3,False)

    #convert the inverted diff to 'color'
    can_c = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    
    #canny edges of the blur
    dst = cv.Canny(blur, 10, 200, None, 3)
    
    cdstP = cv.cvtColor(can, cv.COLOR_GRAY2BGR)
    
    #using predicted hough lines here because it seemed like the results were more useful for what I was trying to do
    #it was giving more results we could at least filter down to something close to accurate    
    linesP = cv.HoughLinesP(can, 1, np.pi / 180, 50, None, 150, 20)
    
    #create an empty array for horizontal both horizontal and vertical lines
    l_h=[]
    l_v=[]

    
    if linesP is not None:
        for i in range(0, len(linesP)):


            l = linesP[i][0]
            
            dxdy =(l[2]-l[0])/(l[3]-l[1]+.001)
            dydx =(l[3]-l[1])/(l[2]-l[0]+.001)
            #multidimensional array new_l with dxdy defined by x2-x1/y2-y1. .001 added to denominator to prevent division by zero
            #will clean up some of the slop like that .001 as we go
            

            #append line into appropriate array
            if abs(dxdy)>1:
                new_l=[dxdy,l[0],l[1],l[2],l[3]]
                l_h.append(new_l)
            else:
                new_l=[dxdy,l[0],l[1],l[2],l[3]]
                l_v.append(new_l)
    
    #sort lines function returns the lines array sent as [[slope],[x1,y1,x2,y2]] sorted in ascending order along the absolute value of the slope.
    #sort lines, horizontal lines
    #l_h = sort_lines(l_h)
    #filtered_h = group_lines(l_h,2)
    max_h, min_h=maxmin(l_h,2)
    max_v, min_v=maxmin(l_v,2)
    
    cv.line(cdstP, (int(max_h[1]), int(max_h[2])), (int(max_h[3]), int(max_h[4])), (255,0,255), 1, cv.LINE_AA)
    cv.line(cdstP, (int(min_h[1]), int(min_h[2])), (int(min_h[3]), int(min_h[4])), (255,0,255), 1, cv.LINE_AA)
            
    #for i in range(0,len(l_h)-1):         
    #    h=l_h[i]
        
        #add lines to the cdstP image
    #    cv.line(cdstP, (int(h[1]), int(h[2])), (int(h[3]), int(h[4])), (255,0,255), 1, cv.LINE_AA)
    l_h=[max_h,min_h]     

    for i in range(len(l_h)):         
        h=l_h[i]
        if h[0]>0:
            pt1 = [int(h[1]-1000*h[0]),int(h[2]-1000)]
            pt2 = [int(h[1]+1000*h[0]),int(h[2]+1000)]
        else:
            pt1 = [int(h[1]-1000*h[0]),int(h[2]-1000)]
            pt2 = [int(h[1]+1000*h[0]),int(h[2]+1000)]
        #add lines to the cdstP image
        cv.line(cdstP, pt1, pt2, (175,150,100), 1, cv.LINE_AA)

    
    l_v=[max_v, min_v]
    
    pt1 = line_intersection(((max_v[2],max_v[1]),(max_v[4],max_v[3])),((max_h[2],max_h[1]),(max_h[4],max_h[3])))
    pt2 = line_intersection(((min_v[2],min_v[1]),(min_v[4],min_v[3])),((max_h[2],max_h[1]),(max_h[4],max_h[3])))
    pt3 = line_intersection(((min_v[2],min_v[1]),(min_v[4],min_v[3])),((min_h[2],min_h[1]),(min_h[4],min_h[3]))) 
    pt4 = line_intersection(((max_v[2],max_v[1]),(max_v[4],max_v[3])),((min_h[2],min_h[1]),(min_h[4],min_h[3])))
    cv.line(cdstP, pt1, pt2, (101,180,105), 3, cv.LINE_AA)
    cv.line(cdstP, pt2, pt3, (101,180,105), 3, cv.LINE_AA)
    cv.line(cdstP, pt3, pt4, (101,180,105), 3, cv.LINE_AA)
    cv.line(cdstP, pt4, pt1, (101,180,105), 3, cv.LINE_AA)


    print(pt1,pt2,pt3,pt4)
    
    for i in range(len(l_v)):
        v=l_v[i]
        dx=1000*v[0]

        
        if v[0] > 0:
            pt1 = [int(v[1]-1000*v[0]),int(v[2]-1000)]
            pt2 = [int(v[1]+1000*v[0]),int(v[2]+1000)]
        else: 
            pt1 = [int(v[1]+1000*v[0]),int(v[2]+1000)]
            pt2 = [int(v[1]-1000*v[0]),int(v[2]-1000)]
            
        
        cv.line(cdstP, pt1, pt2, (101,180,105), 1, cv.LINE_AA)
    
    while True:
        
        cv.imshow('cdstP',cdstP)   
  
        if cv.waitKey(1) == ord('q'):
            break        

vid.release()

cv.destroyAllWindows
