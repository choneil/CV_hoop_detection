import cv2 as cv
import numpy as np
import math
from maxmin  import maxmin
from points import line_intersection
vid = cv.VideoCapture('Hoop_vid.mov')
frame_no=0
#Dimensions of video frame
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

#Variables to resize video frame with the correct ratio
# ratio = width/height
ratio = 16/9
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

    #Lower span of blue in HSV coordinates
    lower_blue1 = np.array([26,59,78])
    upper_blue1 = np.array([126,255,255])

    #Mask the Lower span of blue in the frame. All masked pixels will be white while others are black
    mask1_blue = cv.inRange(hsv, lower_blue1, upper_blue1)

    #Upper span of blue in HSV coordinates
    lower_blue2 = np.array([26,143,164])
    upper_blue2 = np.array([180,255,255])

    #Mask the upper span of blue in the frame. All masked pixels will be white while others are black
    mask2_blue = cv.inRange(hsv, lower_blue2, upper_blue2)

    #Combine the lower and upper red masks
    mask_all_blue = mask1_blue + mask2_blue

    #Morphology using the combined mask as the source parameter, and (7,7) as the kernel
    #Explanation of Morphology commands(OPEN, DILATE, etc.) in [1]
    mask_all_blue = cv.morphologyEx(mask_all_blue, cv.MORPH_OPEN, np.ones((7,7),np.uint8))
    mask_all_blue = cv.morphologyEx(mask_all_blue, cv.MORPH_DILATE, np.ones((7,7),np.uint8))

    #More filters to try to hone in on straight lines
    thresh_blue = cv.threshold(mask_all_blue, 130, 255, cv.THRESH_BINARY)[1]
    blur_blue = cv.GaussianBlur(thresh_blue,(9,9),10,None,1)
    kernel_blue = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    dilate_blue = cv.morphologyEx(blur_blue, cv.MORPH_DILATE, kernel_blue)

    # get absolute difference between dilate and the blurred threshold
    # got better results for my lines doing this rather than just the threshold

    diff_blue = cv.absdiff(dilate_blue, blur_blue)

    #invert diff
    edges_blue = 255 - diff_blue

    #canny edges of the diff 
    can_blue = cv.Canny(diff_blue,100,255,None,3,False)

    #convert the inverted diff to 'color'
    can_c_blue = cv.cvtColor(edges_blue, cv.COLOR_GRAY2BGR)

    #canny edges of the blur
    dst_blue = cv.Canny(blur_blue, 10, 200, None, 3)
    
    cdstP_blue = cv.cvtColor(can_blue, cv.COLOR_GRAY2BGR)

    #using predicted hough lines here because it seemed like the results were more useful for what I was trying to do
    #it was giving more results we could at least filter down to something close to accurate 
    linesP_blue = cv.HoughLinesP(can_blue, 1, np.pi / 180, 50, None, 150, 20)

    #create an empty array for horizontal both horizontal and vertical lines
    l_h_b=[]
    l_v_b=[]


    if linesP_blue is not None:
        for i in range(0, len(linesP_blue)):
            l_b = linesP_blue[i][0]

            dxdy_b =(l_b[2]-l_b[0])/(l_b[3]-l_b[1]+.001)
            dydx_b =(l_b[3]-l_b[1])/(l_b[2]-l_b[0]+.001)
            #multidimensional array new_l with dxdy_r defined by x2-x1/y2-y1. .001 added to denominator to prevent division by zero
            #will clean up some of the slop like that .001 as we go
            
            #append line into appropriate array
            if abs(dxdy_b)>1:
                new_l_b=[dxdy_b,l_b[0],l_b[1],l_b[2],l_b[3]]
                l_h_b.append(new_l_b)
            else:
                new_l_b=[dxdy_b,l_b[0],l_b[1],l_b[2],l_b[3]]
                l_v_b.append(new_l_b)
    #sort lines function returns the lines array sent as [[slope],[x1,y1,x2,y2]] sorted in ascending order along the absolute value of the slope.
    #sort lines, horizontal lines
    #l_h = sort_lines(l_h)
    #filtered_h = group_lines(l_h,2)q
    max_h_b, min_h_b=maxmin(l_h_b,2)
    max_v_b, min_v_b=maxmin(l_v_b,2)

    cv.line(cdstP_blue, (int(max_h_b[1]), int(max_h_b[2])), (int(max_h_b[3]), int(max_h_b[4])), (255,0,255), 1, cv.LINE_AA)
    cv.line(cdstP_blue, (int(min_h_b[1]), int(min_h_b[2])), (int(min_h_b[3]), int(min_h_b[4])), (255,0,255), 1, cv.LINE_AA)

     #for i in range(0,len(l_h)-1):         
    #    h=l_h[i]
        
        #add lines to the cdstP image
    #    cv.line(cdstP, (int(h[1]), int(h[2])), (int(h[3]), int(h[4])), (255,0,255), 1, cv.LINE_AA)   
    l_h_b=[max_h_b,min_h_b]

    for i in range(len(l_h_b)):         
        h_b=l_h_b[i]
        if h_b[0]>0:
            pt1_b = [int(h_b[1]-1000*h_b[0]),int(h_b[2]-1000)]
            pt2_b = [int(h_b[1]+1000*h_b[0]),int(h_b[2]+1000)]
        else:
            pt1_b = [int(h_b[1]-1000*h_b[0]),int(h_b[2]-1000)]
            pt2_b = [int(h_b[1]+1000*h_b[0]),int(h_b[2]+1000)]
        #add lines to the cdstP image
        cv.line(cdstP_blue, pt1_b, pt2_b, (175,150,100), 1, cv.LINE_AA)
    
    l_v_b=[max_v_b, min_v_b]

    pt1_b = line_intersection(((max_v_b[2],max_v_b[1]),(max_v_b[4],max_v_b[3])),((max_h_b[2],max_h_b[1]),(max_h_b[4],max_h_b[3])))
    pt2_b = line_intersection(((min_v_b[2],min_v_b[1]),(min_v_b[4],min_v_b[3])),((max_h_b[2],max_h_b[1]),(max_h_b[4],max_h_b[3])))
    pt3_b = line_intersection(((min_v_b[2],min_v_b[1]),(min_v_b[4],min_v_b[3])),((min_h_b[2],min_h_b[1]),(min_h_b[4],min_h_b[3]))) 
    pt4_b = line_intersection(((max_v_b[2],max_v_b[1]),(max_v_b[4],max_v_b[3])),((min_h_b[2],min_h_b[1]),(min_h_b[4],min_h_b[3])))
    cv.line(cdstP_blue, pt1_b, pt2_b, (255,0,0), 3, cv.LINE_AA)
    cv.line(cdstP_blue, pt2_b, pt3_b, (255,0,0), 3, cv.LINE_AA)
    cv.line(cdstP_blue, pt3_b, pt4_b, (255,0,0), 3, cv.LINE_AA)
    cv.line(cdstP_blue, pt4_b, pt1_b, (255,0,0), 3, cv.LINE_AA)

    print(pt1_b,pt2_b,pt3_b,pt4_b)

    for i in range(len(l_v_b)):
        v_b=l_v_b[i]
        dx_b=1000*v_b[0]

        
        if v_b[0] > 0:
            pt1_b = [int(v_b[1]-1000*v_b[0]),int(v_b[2]-1000)]
            pt2_b = [int(v_b[1]+1000*v_b[0]),int(v_b[2]+1000)]
        else: 
            pt1_b = [int(v_b[1]+1000*v_b[0]),int(v_b[2]+1000)]
            pt2_b = [int(v_b[1]-1000*v_b[0]),int(v_b[2]-1000)]
            
        
        cv.line(cdstP_blue, pt1_b, pt2_b, (101,180,105), 1, cv.LINE_AA)
    print(frame_no)
    frame_no = (frame_no + 1)
    while True:
        cv.imshow('mask',mask_all_blue)
        cv.imshow('cdstP_blue',cdstP_blue) 
        cv.imshow('smallframe', small_frame)
  
        if cv.waitKey(1) == ord('q'):
            break        

vid.release()

cv.destroyAllWindows