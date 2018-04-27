# This is an extension of the project, which uses OPENCV
# to perform realtime detection and identification of faces

from keras.models import model_from_json
from keras.utils import np_utils

import cv2
import numpy as np
import os

from skimage import io
from sklearn.cross_validation import train_test_split

# Functions for LBP
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx, idy]
    except IndexError:
        return default

# use the photos in './CNNdata' to test the model first
# load paths in DatasetPath
DatasetPath = []
for i in os.listdir('./CNNdata'):
    DatasetPath.append(os.path.join('./CNNdata', i))

imageData = []
imageLabels = []

# Read and save photos and labels
for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split("_")[0]) - 1
    imageLabels.append(labelRead)

# split the data to 2 parts, one for training, one for testing
# split randomly, and 90% for training, 10% for testing
X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(imageLabels), train_size=0.9, random_state = 4)
X_test = np.array(X_test)

nb_classes = 4

# change the data form for photos of y_test 
y_test = np.array(y_test)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# for tensorflow backend, it's (nb_of_photo, size, size, channel)
X_test = X_test.reshape(X_test.shape[0], 46, 46, 1)
X_test = X_test.astype('float32')
X_test /= 255

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# Evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
score = loaded_model.evaluate(X_test,Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Using OPENCV for face detection
path = '/Users/abhi/code/lib/python2.7/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')

# then we turn on the camera
cap = cv2.VideoCapture(0)

# Use a boolean variable for starting the face recognition
start = False

while(True):
    ret, img = cap.read()

    if start == True:
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	    )
	    for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

	    try:
		    cropped = gray[y:y + h, x:x + w]
		    result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x
		    
		    transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

		    for x in range(0, len(result)):
			for y in range(0, len(result[0])):
			    center = result[x, y]
			    top_left = get_pixel_else_0(result, x - 1, y - 1)
			    top_up = get_pixel_else_0(result, x, y - 1)
			    top_right = get_pixel_else_0(result, x + 1, y - 1)
			    right = get_pixel_else_0(result, x + 1, y)
			    left = get_pixel_else_0(result, x - 1, y)
			    bottom_left = get_pixel_else_0(result, x - 1, y + 1)
			    bottom_right = get_pixel_else_0(result, x + 1, y + 1)
			    bottom_down = get_pixel_else_0(result, x, y + 1)

			    values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
				                          bottom_down, bottom_left, left])

			    weights = [1, 2, 4, 8, 16, 32, 64, 128]
			    res = 0
			    for a in range(0, len(values)):
				res += weights[a] * values[a]

			    transformed_img.itemset((x, y), res)

		    # we only use the part (1,1) to (46,46) of the result img.
		    # original img: 0-47, after resize: 1-46
		    lbp = transformed_img[1:47, 1:47]

		    cc = np.array(lbp)
		    cc = np.array(cc)

		    cc = cc.reshape(1, 46, 46, 1)
		    cc = cc.astype('float32')

		    cc /= 255

		    predictions = loaded_model.predict(cc)
		    font = cv2.FONT_HERSHEY_SIMPLEX
		    
		    if predictions[0][0] > 0.85:
		        cv2.putText(img,'This is Abhi',(x,y+h+30),font,1,(255, 0, 0),2)
		    elif predictions[0][1] > 0.85:
		        cv2.putText(img,'This is Hampi',(x,y+h+30),font,1,(255, 0, 0),2)
		    elif predictions[0][2] > 0.85:
		        cv2.putText(img,'This is Charu',(x,y+h+30),font,1,(255, 0, 0),2)
		    elif predictions[0][3] > 0.85:
		        cv2.putText(img,'This is Kru',(x,y+h+30),font,1,(255, 0, 0),2)
		    else:
		        cv2.putText(img,'Can\'t be recognized',(x,y+h+30),font,1,(255, 0, 0),2)
	    except:
		pass

    cv2.imshow('img',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(10) & 0xFF == ord('i'):
	start = True
    
    
cap.release()
cv2.destroyAllWindows()


