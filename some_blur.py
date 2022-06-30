import numpy as np



#highpass filter
def highpass_filter():
    filter = np.array( [[0, -1, 0 ],
                       [-1, 4, -1],
                       [0, -1, 0 ]] )
    return filter

#sharpen filter
def sharpen_filter():
    filter= np.array([[0, -1, 0 ],
                      [-1, 5, -1],
                      [0, -1, 0 ]])
    return filter

#3x3 gauss filter with standard deviation = 1
def gauss_filter_1():
    kernel= np.array([[1/16, 2/16, 1/16],
                      [2/16, 4/16, 2/16],
                      [1/16, 2/16, 1/16]])
    return kernel


#median filter
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i+z - indexer < 0 or i+z - indexer > len(data)-1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j+z - indexer < 0 or j+indexer > len(data[0])-1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i+z - indexer][j+k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

# gaussian Kernel
def gaussian_kernel(sigma):
    range= round(sigma*3)
    size = range*2 + 1
    x, y = np.mgrid[0:size+1, 0:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

