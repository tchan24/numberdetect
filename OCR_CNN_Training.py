import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
#import pickle

#############################
path = 'myData'
images = []
classnumber = []
testratio = 0.2
validationratio = 0.2
imagedimensions = (32,32,3)
batchSize = 50
epochsVal = 10
epochSteps = 2000
#############################

mylist = os.listdir(path)
print("Total number of classes detected",len(mylist))
num_of_classes = len(mylist)

print("Importing classes:")
for count in range(0,num_of_classes):
    pic_list = os.listdir(path+"/"+str(count))
    for y in pic_list:
        current_image = cv2.imread(path+"/"+str(count)+"/"+y)
        current_image = cv2.resize(current_image, (imagedimensions[0],imagedimensions[1]))
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

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

Y_train = to_categorical(Y_train, num_of_classes)
Y_test = to_categorical(Y_test, num_of_classes)
Y_validation = to_categorical(Y_validation, num_of_classes)

def myModel():
    num_of_filters = 60
    size_of_filter1 = (5,5)
    size_of_filter2 = (3,3)
    size_of_pool = (2,2)
    num_of_nodes = 500

    model = Sequential()
    #model.add((Conv2D(num_of_filters, size_of_filter1, input_shape = (imagedimensions[0], imagedimensions[1], 1), activation='relu')))
    #model.add((Conv2D(num_of_filters, size_of_filter1, activation='relu')))
    #model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(Input(shape=(imagedimensions[0], imagedimensions[1], 1)))
    model.add(Conv2D(num_of_filters, size_of_filter1, activation='relu'))
    model.add(Conv2D(num_of_filters, size_of_filter1, activation='relu'))
    model.add((Conv2D(num_of_filters//2, size_of_filter2, activation='relu')))
    model.add((Conv2D(num_of_filters//2, size_of_filter2, activation='relu')))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_of_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])    

    return model

model = myModel()
print(model.summary())


history = model.fit(dataGen.flow(X_train, Y_train, batch_size=batchSize), steps_per_epoch=epochSteps, epochs=epochsVal, validation_data=(X_validation, Y_validation), shuffle=True)

# plotting results
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# eval
score = model.evaluate(X_test, Y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#save model
#pickle_out= open("model_trained.p", "wb")
#pickle.dump(model,pickle_out)
#pickle_out.close()
# Save the trained model
model.save("model_trained.keras")