import cv2 as cf
import numpy as np


ld = cf.imread('Photos/group 1.jpg')
cf.imshow('aj',ld)

# translating image means shifting x&y axis of image in any dirn
def translate(ld,x,y) :

    transmatrix = np.float32([[1,0,x],[0,1,y]])         # new_translation_matrix
    dimension = ld.shape[0] , ld.shape[1]

    return cf.warpAffine(ld,transmatrix,dimension)       # applying the transformation_to_image(matrix)

# -x for left and -y for up , y for down
t = translate(ld,-100,200)
cf.imshow('t',t)



# rotating image
def rotate(ld , angleofrot , rotpoint=None ) :

    height , width = ld.shape[:2]      # taking values_upto_2nd_index ,no_needof 2ndas (it will be channel)

    if rotpoint is None :      # considering to rotate from center
        rotpoint= width//2,height//2

    rotmatrix = cf.getRotationMatrix2D(rotpoint,angleofrot,1.0)  # new_matrix , here 3rd param is scale factor 

    return cf.warpAffine(ld,rotmatrix,(width,height))        

rot = rotate(ld,-45)         # rotpoint-not_specified:- ie .centre of image
cf.imshow('ro',rot)


# Resizing image
cf.imshow('u',cf.resize(ld,(300,450),interpolation=cf.INTER_CUBIC))

# Flipping image
cf.imshow('g',cf.flip(ld,1))      # 0 means flip around x 1 means around y,-1 boleto both side

cf.waitKey(0)


