import cv2 as cns
import numpy as np


peop = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_cascade = cns.CascadeClassifier('haar_frontface_classifierl.xml')     #pre-trained classifer to detect faces


face_recog = cns.face.LBPHFaceRecognizer_create()  #instance of opencv's_face_recognisermodel

face_recog.read('face_trained.yml')       # loaded trained model

random_imag = cns.imread(r'C:\Users\impar\Pictures\OpenCV_Course\Face_detection\Faces\val\madonna\4.jpg')     
#r'' used_to_disablethe \ mechanism

gry = cns.cvtColor(random_imag,cns.COLOR_BGR2GRAY)  # returns 3d arr into grayscale2d-arr 
cns.imshow('g',gry)


face_rect = haar_cascade.detectMultiScale(gry,1.1,4)      #for_each_face_found returns 4values


for w,x,y,z in face_rect :

    face_region = gry[x:x+z,w:w+y]      # general_syntax - [ start_row : end_row , start_column : end_column ]

    label , confidence = face_recog.predict(face_region)
    print(f'Label = {peop[label]} with a confidence of {confidence}')

    cns.putText(random_imag , str(peop[label]) , (20,20) , cns.FONT_HERSHEY_SIMPLEX , 1.0 , (0,255,0) , thickness=2)
    cns.rectangle(random_imag , (w,x) , (w+y , x+z) , (0,255,0) , thickness=2)


cns.imshow('detected_face',random_imag)
cns.waitKey(0)

