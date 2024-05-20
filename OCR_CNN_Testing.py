import numpy as np
import cv2
#import pickle
import tensorflow as tf


#############################
width = 640
height = 480
#############################

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

#load model
#pickle_in = open("model_trained.p", "rb")
#model = pickle.load(pickle_in)
model = tf.keras.models.load_model("model_trained.keras")

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, originalimage = cap.read()
    img = np.array(originalimage)
    img = cv2.resize(img, (32, 32))
    img  = preProcess(img)
    #cv2.imshow("Processed image", img)
    img = img.reshape(1, 32, 32, 1)

    #prediction
    #classindex = int(model.predict_classes(img))
    #print(classindex)
    predictions = model.predict(img)
    classindex = np.argmax(predictions)
    print(classindex)

    # Convert the image back to a displayable format
    display_img = img.reshape(32, 32)  # Reshape for display
    display_img = (display_img * 255).astype(np.uint8)  # Convert back to uint8 type

    display_img = cv2.resize(display_img, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Processed image", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release webcam and destroy all opencv
cap.release()
cv2.destroyAllWindows()