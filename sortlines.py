import numpy as np

def sort_lines(lines):
    for i in range(0,len(lines)):
        
        print(lines[i])
    
    for n in range(len(lines)-1,0,-1):
        for i in range(n):
        
            if lines[i][0]>lines[i+1][0]:
                lines[i], lines[i+1]=lines[i+1],lines[i]
    for i in range(len(lines)):

        print(lines[i])




    