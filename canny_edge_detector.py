import numpy as np
import img_function as imf


# gaussian Kernel
def gaussian_kernel(sigma):
    range= round(sigma*3)
    size= range*2 + 1
    x, y= np.mgrid[0:size+1, 0:size+1]
    normal= 1 / (2.0 * np.pi * sigma**2)
    g=  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


# calcolo gradiente e fase
def sobel_filters(img):
    
    Kx= np.array([[-1, 0, 1], 
                  [-2, 0, 2], 
                  [-1, 0, 1]], 
                   np.float32)
    Ky= np.array([[1, 2, 1], 
                  [0, 0, 0], 
                  [-1, -2, -1]], 
                   np.float32)
    Ix= imf.convolve(img, Kx, True)
    Iy= imf.convolve(img, Ky, True)
    G= np.sqrt(Ix**2 + Iy**2)
    G= G / G.max() * 255
    theta= np.arctan2(Iy, Ix)
    
    return (G, theta)


# calcolo NMS (Non Maximum Suppression)
def non_max_suppression(img, D):
    M, N= img.shape
    Z= np.zeros((M,N), dtype=np.int32)
    angle= D * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q= 255
                r= 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q= img[i, j+1]
                    r= img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q= img[i+1, j-1]
                    r= img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q= img[i+1, j]
                    r= img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q= img[i-1, j-1]
                    r= img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j]= img[i,j]
                else:
                    Z[i,j]= 0


            except IndexError as e:
                pass

    return Z


# calcolo Double Threshold
def threshold(im, lowThresholdRatio, highThresholdRatio, strong, weak):
     

    rows= imf.rows_gray(im)
    cols= imf.cols_gray(im)
    res= im
    

    for y in range(0,rows):
        for x in range(0,cols):
            #strong
            if im[x,y] >= highThresholdRatio:
                res[x,y]= strong
            #weak
            elif im[x,y] < highThresholdRatio and im[x,y] >= lowThresholdRatio:
                res[x,y]= weak
            else:
                res[x,y]= 0
            

    return res
                

# Faccio Hysteresis
def hysteresis(img, weak, strong):
    rows= imf.rows_gray(img)
    cols= imf.cols_gray(img)
    
    for i in range(1, cols-1):
        for j in range(1, rows-1):
            if img[i,j] == weak:
                    if (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong):
                        img[i, j] = strong
                    else:
                        img[i, j]= 0

    return img