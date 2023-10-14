import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io

from scipy import ndimage
from PIL import Image

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpeg')
img = color.rgb2gray(img)


def myharris(image, w_size, sigma, k):
    
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
    Ixx = ndimage.gaussian_filter(Ix**2, sigma)
    Ixy = ndimage.gaussian_filter(Iy*Ix, sigma)
    Iyy = ndimage.gaussian_filter(Iy**2, sigma)

    #3 compute sum of product of the derivates at each pixel
    #a 0 padding is added to the derivates to permit to calculate the sum of product
    # of each pixle in function of the window_size
    kernel=np.ones((w_size,w_size))
    
    sxx=convolve2d(Ixx, kernel)
    sxy=convolve2d(Ixy, kernel)
    syy=convolve2d(Iyy, kernel)

    #4 the sum of product of each derivate at each pixel is put together into a matrix
    H_det = sxx * syy - sxy**2
    H_trace = sxx + syy
    R = H_det - k * H_trace**2
    
    return R


def Harris_response(matrix):
    Harris_matrix=matrix.copy()
    Harris_matrix[np.where(matrix<100)]=0
    Harris_matrix[np.where(matrix>=100)]=1
    return Harris_matrix


plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(img)
# 3.2
# Evaluate myharris on the image
sigma=2; k=0.1; w_size=5
R = myharris(img, 5, 2, 0.1)
#R = Harris_response(myharris(img, 5, 0.2, 0.1))
plt.subplot(2,2,2)
plt.title('Corner detection on  image')
plt.imshow(R)
plt.colorbar()

# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
rotated_img=ndimage.rotate(img, angle=45 )
R_rotated=myharris(rotated_img, w_size, sigma, k)
#R_rotated= Harris_response(myharris(rotated_img, w_size, sigma, k))
plt.subplot(2,2,3)
plt.title('Corner detection on rotated  image')
plt.imshow(R_rotated)
plt.colorbar()
# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
arr_resized = np.array(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)))
R_scaled =  myharris(arr_resized, w_size, sigma, k)
#R_scaled = Harris_response( myharris(arr_resized, w_size, sigma, k))
plt.subplot(2,2,4)
plt.title('Corner detection on scaled image')    
plt.imshow(R_scaled)
plt.colorbar()
plt.show()