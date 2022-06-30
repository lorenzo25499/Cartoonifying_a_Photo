import numpy as np
import math as m

#numero righe img rgb
def rows(img):
    c,r,ch= img.shape
    return r

#numero colonne img rgb
def cols(img):
    c,r,ch= img.shape
    return c
#numero canali img rgb
def channels(img):
    c,r,ch= img.shape
    return ch

#numero righe img gray (ch=1)
def rows_gray(img):
    c,r= img.shape
    return r

#numero colonne img gray (ch=1)
def cols_gray(img):
    c,r= img.shape
    return c

#convert rgb to grayscale
def rgb2gray(rgb):
    r, g, b= rgb[:,:,0] , rgb[:,:,1] , rgb[:,:,2]  
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#dilation with 2x2 kernel 
def dilation_2x2(img):
    rows= img.shape[0]
    cols= img.shape[1]
    img_res= np.copy(img);
    for i in range(rows):
        for j in range(cols):
            if img[i,j] == 255:
                if j+1 < cols:
                    img_res[i,j+1]= 255
                if i+1 < rows:
                    img_res[i+1,j]= 255
                if i+1 < rows and j+1 < cols:
                    img_res[i+1,j+1]= 255


    return img_res

#l1_normalize
def l1_normalize(img):
    for k in range(img.shape[2]):
        sum = 0;
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                sum+= img[x,y,k]
        
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img[x,y,k] /= sum


#down_sampled_one_channel
def calcolo_media_pixel_un_canale(img, row, col, size):
    somma=0
    for x in range(size*row, size*row+size):
        for y in range(size*col, size*col+size):
            if x >= img.shape[1]:
                x= img.shape[1]-size
            if y >= img.shape[0]:
                y= img.shape[0]-size
            somma += img[y,x]
    n_pixel= size*size    
    return somma // n_pixel

def down_sampled_one_channel(img,size):
    res= np.zeros((img.shape[0]//size, img.shape[1]//size))
    for row in range(res.shape[1]):
        for col in range(res.shape[0]):
            res[col,row]= calcolo_media_pixel_un_canale(img, row, col, size)
    return res


#resize: bilinear
def pixel_bilinear(img,x,y,c):
    x1= m.floor(x)
    x2= m.ceil(x)
    y1= m.floor(y)
    y2= m.ceil(y)

    if x1 >= img.shape[0]:
        x1= img.shape[0]-1
    if y1 >= img.shape[1]:
        y1= img.shape[1]-1
    if y2 >= img.shape[1]:
        y2= img.shape[1]-1
    if x2 >= img.shape[0]:
        x2= img.shape[0]-1

    v11= img[x1][y1][c]
    v12= img[x1][y2][c]
    v21= img[x2][y1][c]
    v22= img[x2][y2][c]

    v_lin1= (x-x1)*v21 + (x2-x)*v11
    v_lin2= (x-x1)*v22 + (x2-x)*v12
    pixel= (y-y1)*v_lin2 + (y2-y)*v_lin1

    return pixel

def bilinear_resize(img, w, h):
    
    ret= np.zeros((w,h,img.shape[2]))
    for i in range(w):
        for j in range(h):
            x= (i+0.5)*(img.shape[0]/w)-0.5
            y= (j+0.5)*(img.shape[1]/h)-0.5
            for k in range(img.shape[2]):
                ch_val= pixel_bilinear(img,x,y,k)
                ret[i][j][k]= ch_val
    return ret


#quantize color
def change_pixel(pixel, a):
    new_pixel= m.floor(pixel/a) * a
    return new_pixel

def quantize_color(img, a):
    ret= np.zeros(img.shape)
    for c in range(img.shape[2]):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                ret[x,y,c]= change_pixel(img[x,y,c],a)
    return ret


#recombine 
def recombine(img_color, img_dilation):
    img_res= np.copy(img_color)
    for c in range(img_color.shape[2]):
        for x in range(img_color.shape[0]):
            for y in range(img_color.shape[1]):
                if img_dilation[x,y]==255:
                    img_res[x-10,y-8,c]= 0

    return img_res

#convolution
def convolve(img, filter, preserve):
    filter_offset = filter.shape[0]// 2;
    if preserve:
        ret = np.zeros(img.shape)
    else:
        ret= np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if len(img.shape)==2:
                sum=0    
                for l in range(0,filter.shape[0]):
                    for m in range(0,filter.shape[1]):
                        a = clamped_pixel_gray(img, i - l, j - m);
                        b = filter[l][m]
                        sum += a * b
                ret[i, j] = sum

            elif len(img.shape)==3:
                if (preserve): 
                    for k in range(img.shape[2]):
                        sum = 0
                        for l in range(-filter_offset,filter_offset+1):
                            for m in range(-filter_offset,filter_offset+1):
                                a = clamped_pixel(img, i - l, j - m, k);
                                b = filter[filter_offset - l][filter_offset - m]
                                sum += a * b
                            
                        ret[i, j, k] = sum
                    
                else:
                    sum = 0
                    for l in range(-filter_offset,filter_offset+1):
                        for m in range(-filter_offset,filter_offset+1):
                            for k in range(img.shape[2]):
                                a = clamped_pixel(img, i - l, j - m, k)
                                b = filter[filter_offset - l][filter_offset - m]
                                sum += a * b;
                            
                        
                    ret[i, j] = sum;
    return ret;


#clamped_pixel_gray
def clamped_pixel_gray(img, x, y):
    if x<0:
        x=0
    if y<0:
        y=0
    if x>= img.shape[0]:
        x= img.shape[0]-1
    if y>= img.shape[1]:
        y= img.shape[1]-1
    return img[int(x)][int(y)]

#clamped_pixel
def clamped_pixel(img,x,y,c):
    if x<0:
        x=0
    if y<0:
        y=0
    if c<0:
        c=0
    if x>= img.shape[0]:
        x= img.shape[0]-1
    if y>= img.shape[1]:
        y= img.shape[1]-1
    if c>= img.shape[2]:
        c= img.shape[2]-1
    return img[int(x)][int(y)][int(c)]