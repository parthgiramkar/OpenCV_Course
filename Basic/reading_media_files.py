import cv2 as cn

''' read the image_path and return image as matrix of pixel to store in computer memory

img = cn.imread('Photos/cat_large.jpg')

cn.imshow('Cat',img)     # display the window_name with the o/p_matrix
cn.waitKey(0) '''# waits for inf_time


# for Reading videos
capt = cn.VideoCapture('Videos/kitten.mp4')          # can aaccepts integer_also-0 gen_For webcam(closed)

while True :

    istrue , frame = capt.read()   # returns boolean and frame_matrix
    cn.imshow('video' , frame) 

    if cn.waitKey(20) & 0xFF==ord('b') :
        break

capt.release()
cn.destroyAllWindows()




'''
the comp reads the grid of_pixels ,then uses numpy array(i.e -in matrix format) to store into comp_memory
for grayscale(b&w), image the numpy array is represented as (height, width)
for bgr image(by_default_opencv_readit) Color (height, width, channel(i.e-3) ) 3are the std.colors that are used_here
also , in images -all pixel data consists of numbers of the same type (uint8).
'''



