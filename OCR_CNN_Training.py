import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#############################
path = 'myData'
images = []
classnumber = []
testratio = 0.2
validationratio = 0.2
#############################

mylist = os.listdir(path)
print("Total number of classes detected",len(mylist))
num_of_classes = len(mylist)

print("Importing classes:")
for count in range(0,num_of_classes):
    pic_list = os.listdir(path+"/"+str(count))
    for y in pic_list:
        current_image = cv2.imread(path+"/"+str(count)+"/"+y)
        current_image = cv2.resize(current_image, (32,32))
        images.append(current_image)
        classnumber.append(count)
    print(count, end=" ")
    count += 1
print(" ")
print("Total images in image list:", len(images))
print("Total IDs in classnumber list:", len(classnumber))

images = np.array(images)
classnumber = np.array(classnumber)

print(images.shape)
#print(classnumber.shape)

# splitting data

X_train, X_test, Y_train, Y_test = train_test_split(images, classnumber, test_size=testratio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validationratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

num_of_samples = []
for x in range(0, num_of_classes):
    #print(len(np.where(Y_train==x)[0]))
    num_of_samples.append(len(np.where(Y_train==x)[0]))
print(num_of_samples)

plt.figure(figsize=(10,5))
plt.bar(range(0, num_of_classes), num_of_samples)
plt.title("Num of images per class")
plt.xlabel("Class ID")
plt.ylabel("number of images")
plt.show()

#equalize images
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcess(X_train[30])
# img = cv2.resize(img, (300,300))
# cv2.imshow("Preprocessed image", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcess, X_train)))
X_test = np.array(list(map(preProcess, X_test)))
X_validation = np.array(list(map(preProcess, X_validation)))

#adding depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

