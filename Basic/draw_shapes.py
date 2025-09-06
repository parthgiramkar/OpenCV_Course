import cv2 as cnf
import numpy as no


# creating_blank_image , np.zeros- thearray's numb are set_to_0 - which_is_for_blackimage(0,0,0)
img_blank = no.zeros((500,500,3) , dtype='uint8')    # dtype_for_image(all_pixels_areofsamedtype) and 3para- for widht,height,color :-color i.e-channels
cnf.imshow('k',img_blank)

'''
# 1. painting image_of_certain_color

#img_blank[:] = 0,255,0     #select all pixels to colorgreen
img_blank[200:300 ,300:400] = 0,0,255        # sliced rows (height)-200-299 and columns (width)-300-399
cnf.imshow('color',img_blank)

# 2. Drawing the rectangle
#cnf.rectangle(img_blank,(0,0),(300,600), (255,0,0)  , thickness=5)         # out-of-dimens-still-runs and -1 points to cnf.FILLED 
cnf.rectangle(img_blank,(0,0),(250,600), (255,0,0)  , thickness=cnf.FILLED)  # (img_blank.shape[0]//2 , img_blank.shape[1]//2 ) width,height

cnf.imshow('rec',img_blank)


# 3. Drawing the circle
cnf.circle(img_blank,(img_blank.shape[1]//2 , img_blank.shape[0]//2) ,125 , (0,255,0) , thickness = 2) 
cnf.imshow('circel',img_blank)


# 4. Drawing the line
cnf.line(img_blank,(250,0),(300,250),(255,255,255), thickness=5)
cnf.imshow('line',img_blank)
'''

# Text on_an_image                                    # the coordinate from where_the_text_will_start
cnf.putText(img_blank,'Namaste, my Name is Jason Roy ' , (0,200) , cnf.FONT_HERSHEY_TRIPLEX, 1 , (0,255,0) , thickness=1)
cnf.imshow('text',img_blank)

cnf.waitKey(0)


# dtype='uint8' set array numb's data_type to 8-bit_unsigned_integer(0-255) standard_datatype_forimages  


