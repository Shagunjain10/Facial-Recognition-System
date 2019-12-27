
#TRAINING A MODEL
print("shagun")


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
