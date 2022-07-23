import numpy as np
import cv2 as cv
import sys


def merge(b,g,r):
    ret= np.zeros(img_color.shape)
    
    ret[:,:,0]= b
    ret[:,:,1]= g
    ret[:,:,2]= r
    return ret

img_color= cv.imread(sys.argv[1])
b,g,r= cv.split(img_color)

cv.imwrite("b.jpg", b)

cv.imwrite("g.jpg", g)

cv.imwrite("r.jpg", r)


ret= merge(b,g,r)
cv.imwrite("ret.jpg",ret)
