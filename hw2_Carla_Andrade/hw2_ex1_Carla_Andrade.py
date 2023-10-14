""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb

img = plt.imread('cat.jpg').astype(np.float32)

# 1.1
import numpy as np
def boxfilter(n):
    # this function returns a box filter of size nxn


    box_filter=np.arange(n**2, dtype=np.float32).reshape(n,n)
    box_filter=box_filter/box_filter.sum()
    

    return box_filter



def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    #1D ilter
    if len(np.shape(filt))==1:
        m=n=np.shape(filt)[0]
        filt=np.flip(filt)
    
    #2D filter
    else:
        m,n=np.shape(filt)
        filt=np.flip(np.flip(filt,1),0)
    #if  filter is not sqaure the padding is different
    padding=((m)//2, (n)//2)
    
    output=np.full_like(image,0)
    #RGB image
    if len(image.shape)==3:

        k,l,d=np.shape(image)
        
        padded_matrix=np.pad(image, [padding, padding,(0,0)], mode='constant')
        for i in range(k):
            for j in range(l):
                for dim in range(d):
                
                    output[i,j,dim]=np.sum(padded_matrix[i:i+m, j:j+n,dim] * filt)
        filtered_img=np.pad(output, [padding, padding,(0,0)], mode='constant')

    #2D image    
    elif len(image.shape)==2:
       
        k,l=np.shape(image)
        
        padded_matrix=np.pad(image, padding, mode='constant')
    
        for i in range(k):
            for j in range(l):
        
                output[i,j]=np.sum(padded_matrix[i:i+m, j:j+n] * filt)
        

        filtered_img=np.pad(output, padding, mode='constant')
    #1D image
    else:

        padded_matrix=np.pad(image, padding, mode='constant')
        k=np.shape(image)[0]
        for i in range(k):
            output[i]=np.sum(np.sum(padded_matrix[i:i+m] * filt))

        filtered_img=np.pad(output, padding, mode='constant')



    return filtered_img


# 1.3 after 1.4


# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###
    if filter_length%2==0:
        x=np.arange(-filter_length//2,filter_length//2+1)
    else:
        x=np.arange(-filter_length//2,filter_length//2)
 
    
    

    gauss_filter=np.exp((-x**2)/(2*(sigma**2)))

    return gauss_filter/gauss_filter.sum()



if __name__ == '__main__':

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('original image')
    # 1.3
    # create a boxfilter of size 10 and convolve this filter with your image - show the result
    bsize = 11
    filt=boxfilter(bsize)
    image_conv=myconv2(img,filt)
    plt.subplot(1,2,2)
    plt.imshow(image_conv/255)
    plt.axis('off')
    plt.title('Convulated image')
    plt.show()

    #1.4  of a 1D filter with sigma=2
    gauss1d(2)



