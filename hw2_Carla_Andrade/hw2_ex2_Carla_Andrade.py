import matplotlib.pyplot as plt
import numpy as np
from hw2_ex1_Carla_Andrade import myconv2, gauss1d
from scipy import ndimage

img = plt.imread('cat.jpg').astype(np.float32)
plt.rcParams['image.cmap'] = 'gray'

# 2.1
# Gradients
# define a derivative operator
dx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
dy = dx.T

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1

gdx = myconv2(gauss1d(sigma),dx)
gdy = myconv2(gauss1d(sigma),dy)


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @dx     : gradient along x axis
    # @dy     : gradient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    #filter noise
    image=ndimage.gaussian_filter(image, sigma=1)
    Ix=myconv2(image,dx)
    Iy=myconv2(image,dy)

    grad_mag_image= np.sqrt(Ix**2+Iy**2)
    grad_dir_image=np.arctan2(dy,dx)
    grad_mag_image=grad_mag_image/grad_mag_image.max()*255

    

    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')

plt.show()