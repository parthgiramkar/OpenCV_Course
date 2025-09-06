import cv2 as cv

#img = cv.imread('Photos/cat_large.jpg')

# function for resizing imag's and videos
def rescale(frame , scale=0.15) :

    width =  int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = width,height    # new_dimensions
    return cv.resize(frame,dimensions , interpolation=cv.INTER_AREA)

#rescale_img = rescale(img)        # bydefault its 15% scale factor
#cv.imshow('C',rescale_img)         

'''function for live video resoln
def change_resoln(width,height) :

    capture.set(3,width)
    capture.set(4,height)            '''


capture = cv.VideoCapture('Videos/dog.mp4')
while True :

    istrue,frame=capture.read()

    cv.imshow('v',rescale(frame,scale=0.50))          # set the newrescale value

    if cv.waitKey(20) & 0xFF==ord('b') :
        break

capture.release()
cv.destroyAllWindows()

