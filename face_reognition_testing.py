#TRAINING A MODEL


import cv2
import numpy as np
from os import listdir           # to fetch file  listdir is a class in os module used to fetch the data from any directory
from os.path import isfile, join # to g

data_path = 'C:/New2/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]  # # listdir me loop lga he f variable ko fetch krega
#print(onlyfiles)
#print(f)
# f is a variable  in this as isfile give to true it join data_path and f
#os.listdir() method in python is used to get the list of all files and directories in the specified directory. ... Return Type:
#This method returns the list of all files and directories in the specified path.


Training_Data, Labels = [], [] # both are list empty

for i, files in enumerate(onlyfiles):      
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # it read only grayscale image 
    Training_Data.append(np.asarray(images, dtype=np.uint8)) #unsigned int
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()  #linear bianry phase his face

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")


#TESTING A MODEL

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

        return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)


        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

            if cv2.waitKey(1)==13:
                break


cap.release()
cv2.destroyAllWindows()
