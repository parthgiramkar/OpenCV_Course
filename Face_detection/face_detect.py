import cv2 as cf

img = cf.imread('Photos/group 1.jpg')     # saved_the_imginformof_numpyarr(matrix)
cf.imshow('lad',img)

grey = cf.cvtColor(img,cf.COLOR_BGR2GRAY)  # haar_cascade algo isdesigned_toworkon_b&w intensity_info
cf.imshow('k', grey)

haar_cascade = cf.CascadeClassifier('haar_frontface_classifierl.xml')

face_rect = haar_cascade.detectMultiScale(grey , scaleFactor=1.1, minNeighbors=1)  # returns the list of rectangles where it found faces
print("no. of faces found" ,len(face_rect) )



# rectangles around the face 
for w,x,y,z in face_rect :              # w-xcoordinate ,x-ycord ,y-width,z-height
    
    cf.rectangle(grey, (w,x) , (w+y ,x+z) , (0,255,0) , thickness=1) 


cf.imshow('image_detected', grey)

cf.waitKey(0)


# haar_cascade algo has,limitation to correctly identify_allfaces fails_when_no.ofpersons_increases
