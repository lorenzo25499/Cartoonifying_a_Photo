import numpy as np




#bilateral filter && aux functions
def gaussian(pixel, sigma):
    return (1/(2*np.pi*(sigma**2))) * np.exp(-(pixel**2)/(2*(sigma**2)))
    
def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2 + (y1-y2)**2))

def bil_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(np.round(filtered_image))
    return new_image

