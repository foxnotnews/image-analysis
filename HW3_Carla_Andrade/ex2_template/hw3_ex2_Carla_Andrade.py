import matplotlib.pyplot as plt
import numpy as np
import utils as utls

#
# Constants
#

# Change patch_half_size to change the patch size used (patch size is 2 * patch_half_size + 1)
patch_half_size = 10
patchSize = 2 * patch_half_size + 1

# Display results interactively
showResults = True

# Read input image

filename = 'ex2_template/donkey'
#filename = 'ex2_template/tomato'
#filename = 'ex2_template/yacht'

# load image
im_array = plt.imread(filename + '.jpg')
imRows, imCols, imBands = np.shape(im_array)

# load fill region mask
fill_region = plt.imread(filename + '_fill.png')

# load texture mask
texture_region = plt.imread(filename + '_texture.png')

# prepare to display the first 2 subplots of the output
if showResults:
    plt.subplot(1,3,1)
    plt.imshow(im_array)
    plt.title('original Image')

    # create an image with the masked region blacked out, and a rectangle indicating where to
    # fill it from
    im_mask_fill = np.copy(im_array)
    im_mask_fill[np.where(fill_region)] = [0, 0 , 0]
    texture_outline = utls.find_edge(texture_region)
    im_mask_fill[np.where(texture_outline)] = [255, 255, 255]

    # show it
    plt.subplot(1,3,2)
    plt.imshow(im_mask_fill)
    plt.title('Image with masked region and region to take texture from')

#
# Get coordinates for masked region and texture regions
#
fill_indices = fill_region.nonzero()
assert((min(fill_indices[0]) >= patch_half_size) and
        (max(fill_indices[0]) < imRows - patch_half_size) and
        (min(fill_indices[1]) >= patch_half_size) and
        (max(fill_indices[1]) < imCols - patch_half_size)), "Masked region is too close to the edge of the image for this patch size"

texture_indices = texture_region.nonzero()
texture_img = im_array[min(texture_indices[0]):max(texture_indices[0]) + 1,
                        min(texture_indices[1]):max(texture_indices[1]) + 1, :]
assert((texture_img.shape[0] > patchSize) and
        (texture_img.shape[1] > patchSize)), "Texture region is smaller than patch size"

#
# Initialize im_filled for texture synthesis (i.e., set fill pixels to 0)
#

im_filled = im_array.copy()
im_filled[fill_indices] = 0
while (len(fill_indices[0])  > 0):
    print("Number of pixels remaining = ", len(fill_indices[0]) )

    # Set fill_region_edge to pixels on the boundary of the current fill_region
    fill_region_edge = utls.find_edge(fill_region)
    edge_pixels = fill_region_edge.nonzero()

    while(len(edge_pixels[0]) > 0):

        # Pick a random pixel from the fill_region_edge
        random_index=np.random.randint(0, len(edge_pixels[0] ))
        patch_center_i, patch_center_j = np.array(edge_pixels)[:,random_index]


        # Isolate the patch to fill, and its mask
        patch_to_fill = im_filled[patch_center_i- patch_half_size : patch_center_i+ patch_half_size+1, patch_center_j- patch_half_size: patch_center_j+ patch_half_size+1, :]
        patch_mask = fill_region[patch_center_i- patch_half_size : patch_center_i+ patch_half_size+1, patch_center_j- patch_half_size: patch_center_j+ patch_half_size+1]
        #
        # Compute masked SSD of patch_to_fill and texture_img
        #
        ssd_img = utls.compute_ssd(patch_to_fill, patch_mask, texture_img, patch_half_size)
       
        ## Select the best texture patch
        ssdImg = np.sort(np.copy(ssd_img),axis=None)
        
        #ssdValue = ssdImg[min(int(round(abs(random.gauss(0,1)))),np.size(ssdImg)-1)]
        #ssdValue=ssdImg[int(round(abs(random.gauss(0,1))))]
        
        
        
        ssdIndex = np.nonzero(ssd_img==ssdImg[0])
        #print(ssdIndex)
        
        selected_center_i = ssdIndex[0][0] + patch_half_size
        selected_center_j= ssdIndex[1][0] + patch_half_size
#
        ##
        ## Copy patch into masked region
        ##
        im_filled = utls.copy_patch(im_filled, patch_mask, texture_img, patch_center_i, patch_center_j, selected_center_j, selected_center_j, patch_half_size)
#
        ## Update fill_region_edge and fill_region by removing locations that overlapped the patch
        fill_region_edge[patch_center_i- patch_half_size: patch_center_i+ patch_half_size+1,patch_center_j- patch_half_size: patch_center_j+ patch_half_size+1]=0
        fill_region[patch_center_i- patch_half_size: patch_center_i+ patch_half_size+1,patch_center_j- patch_half_size: patch_center_j+ patch_half_size+1]=0
        
        ## update edge pixels
        edge_pixels = fill_region_edge.nonzero()
        print(len(edge_pixels[0]))

    fill_indices = fill_region.nonzero()
#
# Output results
#
if showResults:
    plt.subplot(1,3,3)
    plt.imshow(im_filled)
    plt.title('Filled Image')
    plt.show()
