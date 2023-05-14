
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from sklearn import preprocessing
from keras.utils import to_categorical
import matplotlib.pyplot as plt

Images = []
Labels = [] 

for path in glob.glob("./asl_dataset/*"):
    #print(path)
    label = path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(path,"*.jpeg")):
        #print(img_path)
        img = cv2.imread(img_path)       
        img = cv2.resize(img, (192, 256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img/255.
        
        Images.append(img)
        Labels.append(label)
        
Images = np.array(Images)
Labels = np.array(Labels)

le = preprocessing.LabelEncoder()
le.fit(Labels)
Labels_encoded = le.transform(Labels)
Labels_one_hot = to_categorical(Labels_encoded)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(Images, Labels_one_hot, test_size=0.2)

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('Y_train shape: ', Y_train.shape)
print('Y_test shape: ', Y_test.shape)

def Feature_Extractor_VGG16(input_shape):
    
    base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    for layer in base.layers:
        layer.trainable = False
    
    output = base.output
    model = Model(inputs=base.input, outputs=output)

    return model

model = Feature_Extractor_VGG16((256,192,3))
model.summary()


extractor = model.predict(X_train)
features = extractor.reshape(extractor.shape[0], -1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
RF_model.fit(features, Y_train)
history = RF_model.fit(features, Y_train)

import joblib
joblib.dump(RF_model, "./VGG16_RandomForest.joblib")



