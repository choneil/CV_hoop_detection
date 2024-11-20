import numpy

def maxmin(list, value):

    for i in range(len(list)):
        
        coord = list[i][value]
        max = list[i]
        min = list[i]
        for j in range(len(list)):
            
            if list[j][value] > max[value]:
                max = list[j]
            if list[j][value] < min[value]:
                min = list[j]


    return(max,min)