import cv2 as cv
import numpy as np

oh=cv.imread('Photos/lady.jpg')
cv.imshow('lady',oh)


cv.imshow('a',cv.cvtColor(oh,cv.COLOR_BGR2GRAY))   # conv_to_b&w(img)_asthereare_onlytwocolors_to_sodetectionbecomes_easier

cv.imshow('a2',cv.Canny(oh,125,100))    


# contours are the outlines or the boundaries of an object in an image.

contour , hierarchy= cv.findContours(cv.Canny(oh,125,100) , cv.RETR_LIST , cv.CHAIN_APPROX_NONE)
print(len(contour) , "no.of contours are present")

#getting blank image
blank = np.zeros(oh.shape , dtype='uint8')
cv.imshow('blank',blank)

dra = cv.drawContours(blank,contour,-1,(0,255,0),1)   # -1 means draw all contours
cv.imshow('f',dra)

cv.waitKey(0)


