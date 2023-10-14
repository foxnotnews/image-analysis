import glob
import os
import numpy as np
import matplotlib.pyplot as plt

#function: pass through the image row by row , top to bottom and left to right.
# if the pixel x has 8 neighbours (q) it will compute thw L1 (Manathan distance) between  x and q
def chamfer_distance(matrix,NP, is_BR=False):
    width,height=matrix.shape
    #don't care about borders because they don't have 8 neighbors. For AL start top to bottom, left to right.
    x_range=range(1,width-1)
    y_range=range(1,height-1)
    #read the matrix from bottom to top, right to left and again don't care about the borders if it's B
    if is_BR:
        x_range=range(width-2,0,-1)
        y_range=range(height-2,0,-1)
    for x in x_range:
        for y in y_range:
            #dx,dy corrsespond to the difference of the corrdinates  between x  and current  q. create a list of distance for either AL ot BR nieghbors. FOr each q neighbor of x calculate 
            #distance_map(x).
            distance_map_x=[((abs(dx) + abs(dy))+matrix[dx+x,dy+y]) for dx,dy in NP]
            distance_map[x,y]=min(distance_map_x)
    
    return distance_map
#difference of the corrdinates  between x  and neighbor  q
Al=[(-1,-1),(0,-1),(1,-1),(-1,0),(0,0)]
BR=[(1,1),(0,1),(-1,1),(1,0),(0,0)]

fig = plt.figure(figsize=(10, 5))
# load shapes
shapes = glob.glob(os.path.join('templates/shapes', '*.png'))
for i, shape in enumerate(shapes):
    # load the edge map
    edge_map = plt.imread(shape)

    # caclulate distance map
    # distance_map: array_like, same size as edge_map with 0 for edges and inf otherwise
    distance_map=np.full_like(edge_map,np.inf)
    distance_map[np.where(edge_map==1)]=0

    #apply chamfer distance for AL and BR neighbors
    distance_map=chamfer_distance(distance_map,Al)
    distance_map=chamfer_distance(distance_map,BR, is_BR=True)

    # the top row of the plots should be the edge maps, and on the bottom the corresponding distance maps
    k, l = i+1, i+len(shapes)+ 1
    plt.subplot(2, len(shapes), k)
    plt.imshow(edge_map, cmap='gray')
    plt.subplot(2, len(shapes), l)
    plt.imshow(distance_map,cmap='gray')

plt.suptitle("Chamfer distance")
plt.show()
