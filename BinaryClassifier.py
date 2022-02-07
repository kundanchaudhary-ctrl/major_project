from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier=Sequential()
#step1-convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
#step2-Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding a second convolutional layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step3-Flattening
classifier.add(Flatten())
#step4-Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part2:Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('C:/Users/KIIT/Pictures/train_set',
target_size=(64,64),
batch_size=32,
class_mode='binary')
test_set=test_datagen.flow_from_directory('C:/Users/KIIT/Pictures/test_set', 
target_size=(64,64),
batch_size=32,
class_mode='binary')
classifier.fit_generator(training_set,
steps_per_epoch=200,
epochs=10,
validation_data=training_set,
validation_steps=100)

import numpy as np
from keras.preprocessing import image

def predict_images(test_image):
    test_image=image.load_img(test_image,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    print(result)
    training_set.class_indices
    if result[0][0]<=0.5:
        prediction='cheque'
    else:
        prediction='random'
    return prediction

#to save and access the model
import pickle
classifier.save("binary_classifier")
pickle.dump(classifier,open("binary_classifier_model",'wb'))
# to nload the model
import pickle
classifier=pickle.load(open("binary_classifier_model",'rb'))

import glob
list_image=[]
for item in glob.glob('C:/Users/KIIT/Pictures/testing/*.PNG'):
    score=predict_images(item)
    list_image.append(score)
    
    
    
    
    
    
    
                                          






