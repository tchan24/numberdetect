import numpy as np
import cv2
import os

images = []
classnumber = []
path = 'myData'

mylist = os.listdir(path)
print("Total number of classes detected",len(mylist))
num_of_classes = len(mylist)

print("Importing classes:")
for x in range(0,num_of_classes):
    pic_list = os.listdir(path+"/"+str(x))
    for y in pic_list:
        current_image = cv2.imread(path+"/"+str(x)+"/"+y)
        current_image = cv2.resize(current_image, (32,32))
        images.append(current_image)
        classnumber.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
classnumber = np.array(classnumber)

print(images.shape)
print(classnumber.shape)