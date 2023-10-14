""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve
from skimage import color, io
from hw2_ex1_Carla_Andrade import *

from scipy import ndimage
from PIL import Image

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpeg')
img = color.rgb2gray(img)

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k, filter_length=10):
    
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    #1 compute x,y derivates
    sobel=np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Ix = convolve(image, sobel)
    Iy = convolve(image, sobel.T)

    #2 compute product of derivates and apply an guassian filter
    #Ixx = ndimage.gaussian_filter(Ix**2, sigma)
    #Ixy = ndimage.gaussian_filter(Iy*Ix, sigma)
    #Iyy = ndimage.gaussian_filter(Iy**2, sigma)
    Ixx = myconv2(Ix**2, gauss1d(sigma=sigma, filter_length=filter_length))
    Ixy = myconv2(Ix*Iy, gauss1d(sigma=sigma, filter_length=filter_length))
    Iyy = myconv2(Iy**2, gauss1d(sigma=sigma, filter_length=filter_length))


    #3 compute sum of product of the derivates at each pixel
    height,width=image.shape
    #a 0 padding is added to the derivates to permit to calculate the sum of product
    # of each pixle in function of the window_size
    padding=w_size//2
    Ixx_pad=np.pad(Ixx, padding)
    Ixy_pad=np.pad(Ixy, padding)
    Iyy_pad=np.pad(Iyy, padding)
    
    R=image.copy()
    for row in range(height):
        for col in range(width):
            sxx=np.sum(Ixx_pad[row: row+w_size, col: col+w_size])
            sxy=np.sum(Ixy_pad[row: row+w_size, col: col+w_size])
            syy=np.sum(Iyy_pad[row: row+w_size, col: col+w_size])
            #4 the sum of product of each derivate at each pixel is put together into a matrix
            H=np.array([[sxx,sxy],[sxy,syy]])
            #5 R is cacluated
            R[row,col]=np.linalg.det(H)-k*(np.trace(H)**2)
    

    

    return R

if __name__ == '__main__':
    plt.subplot(2,2,1)
    plt.title('Original image')
    plt.imshow(img)

    # 3.2
    # Evaluate myharris on the image
    sigma=.2; k=0.1; w_size=5
    R = myharris(img, 5, 0.2, 0.1)
    plt.subplot(2,2,2)
    plt.title('Corner detection on  image')
    plt.imshow(R)
    plt.colorbar()



    # 3.3
    # Repeat with rotated image by 45 degrees
    # HINT: Use scipy.ndimage.rotate() function
    rotated_img=ndimage.rotate(img, angle=45 )
    R_rotated =myharris(rotated_img, w_size, sigma, k)
    plt.subplot(2,2,3)
    plt.title('Corner detection on rotated  image')
    plt.imshow(R_rotated)
    plt.colorbar()



    # 3.4
    # Repeat with downscaled image by a factor of half
    # HINT: Use scipy.misc.imresize() function
    arr_resized = np.array(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)))
    R_scaled =  myharris(arr_resized, w_size, sigma, k)
    plt.subplot(2,2,4)
    plt.title('Corner detection on scaled image')    
    plt.imshow(R_scaled)
    plt.colorbar()
    plt.show()
