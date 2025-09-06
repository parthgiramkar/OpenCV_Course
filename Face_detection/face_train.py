import cv2 as cn
import numpy as np
import os                      #to interact_with_filesystems like (reading directories_or_filepaths)

peop = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# p = []

# for i in os.listdir(r'C:\Users\impar\Pictures\OpenCv Course\Face_detection\Faces\train') :

#     p.append(i)


labels = []         # store_labels_of_the_detected_faces
features = []          # will_store_imagedata_ofthe_detected_faces



haar_cascade = cn.CascadeClassifier('haar_frontface_classifierl.xml')     #pre-trained classifer to detect faces

dir = r'C:\Users\impar\Pictures\OpenCV_Course\Face_detection\Faces\train'
def train_data() :

    for i in peop :            #iterate every folder of train_data

        path=os.path.join(dir,i)         # path to each person's folder
        label=peop.index(i)       # label to each folder

        # inside the folder consider_every_image
        for img in os.listdir(path) :
        

            image_path = os.path.join(path,img)

            image = cn.imread(image_path)
            grey = cn.cvtColor(image,cn.COLOR_BGR2GRAY)        # as the haar_cascade algo is designed to work on b&w intensity info

            face_rect = haar_cascade.detectMultiScale(grey , scaleFactor=1.1, minNeighbors=4)    # for each_facefound return list_of_rect

            for (w,x,y,z) in face_rect :

                face_region = grey[x:x+z , w:w+y]         #cropping the face_image general_syntax - [ start_row : end_row , start_column : end_column ] 

                features.append(face_region)
                labels.append(label)


#print(p)

train_data()
print(len(features))
print(len(labels))

#conv to numpy array as .train requires numpyas_input
features = np.array(features , dtype='object')   # dtype=object bcoz the images are of different sizes,which numpy_arr_dontwork_with

labels=np.array(labels)

face_recog = cn.face.LBPHFaceRecognizer_create()  #uses Local Binary Patterns Histograms (LBPH) algorithm

#training the face recogniser on the features_and_label_list

face_recog.train(features , labels)          #learns the unique patterns for each person

face_recog.save('face_trained.yml')       # saves_the_trained_model

np.save('features.npy' , features)        # also,saving_the_processed_numpy_array's
np.save('labels.npy' , labels)


