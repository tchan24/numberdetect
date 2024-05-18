import numpy as np
import cv2
import os

path = 'myData'

mylist = os.listdir(path)
print(len(mylist))
noOfClasses = len(mylist)

for x in range(0,noOfClasses):
    pass