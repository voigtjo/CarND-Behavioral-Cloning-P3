import csv
import keras.models
import cv2
import numpy as np

lines = []
for file in [ '../data/driving_log.csv']:
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
images = []
angles = []
correction = 0.25
for line in lines:
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]

    originalCenter = cv2.imread(center_path)
    center = cv2.cvtColor(originalCenter, cv2.COLOR_BGR2RGB)

    originalLeft = cv2.imread(left_path)
    left = cv2.cvtColor(originalLeft, cv2.COLOR_BGR2RGB)

    originalRight = cv2.imread(right_path)
    right = cv2.cvtColor(originalRight, cv2.COLOR_BGR2RGB)
    

    center_angle = float(line[3])
    left_angle = center_angle + correction
    right_angle = center_angle - correction

    images.append(center)
    angles.append(center_angle)

    images.append(left)
    angles.append(left_angle)

    images.append(right)
    angles.append(right_angle)

augmented_images = []
augmented_angles = []

for image, angle in zip(images, angles):
    augmented_images.append(np.fliplr(image))
    augmented_angles.append(-angle)

images.extend(augmented_images)
angles.extend(augmented_angles)
    
X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

IMAGE_ROW = 160
IMAGE_COL = 320
IMAGE_CH = 3
input_shape = (IMAGE_ROW, IMAGE_COL, IMAGE_CH)

def createModel():
    model = Sequential()
    model.add(Lambda(lambda x : (x/255.0) -0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape = input_shape))
    return model

def model_nvidia():
    model = createModel()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = model_nvidia()
model.compile(loss='mse', optimizer=Adam(lr=0.001))
history = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 2)
model.save('model.h5')

plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss' ], loc='upper left')
plt.show()


