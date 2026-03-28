import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
labels = []

path = r"D:\myPython\TrafficSignProject\Train"
classes = 43

print("Loading images...")

for i in range(classes):
    folder = os.path.join(path, str(i))
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (32,32))
        data.append(image)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

data = data / 255.0
labels = to_categorical(labels, classes)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")
model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

model.save("traffic_model.h5")

print("Model saved!")