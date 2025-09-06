import cv2 as cs
import matplotlib.pyplot as plt

# be default Opencv uses BGR color space

pic = cs.imread('Photos/group 2.jpg')
cs.imshow('grpou',pic)

plt.imshow(pic)          # but plt works in rgb format it needs to be conv_into rgb to correctly showcase the imag
plt.show()

# BGR convert to greyscale
cs.imshow('gry',cs.cvtColor(pic,cs.COLOR_BGR2GRAY) )

# to hsv - hue saturation_value
cs.imshow('g',cs.cvtColor(pic,cs.COLOR_BGR2HSV))

# to LAB
cs.imshow('l',cs.cvtColor(pic,cs.COLOR_BGR2LAB))


cs.waitKey(0)
