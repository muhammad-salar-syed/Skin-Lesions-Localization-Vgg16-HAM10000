
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
import numpy as np
import scipy
from matplotlib.patches import Rectangle 
from skimage.feature.peak import peak_local_max

datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_generator = datagen.flow_from_directory(
    './Data/',
    target_size=(256,384),
    batch_size=50,
    class_mode='categorical',
    subset='training') 

validation_generator = datagen.flow_from_directory(
    './Data/', 
    target_size=(256,384),
    batch_size=50,
    class_mode='categorical',
    subset='validation')

x, y = next(validation_generator)

def Vgg16_GP_feature(input_shape):
    
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = input_shape)

    for layer in vgg.layers[:-5]:    # Set block5 trainable, all others as non-trainable
        layer.trainable = False 

    x = vgg.output
    x = GlobalAveragePooling2D()(x)  
    x = Dense(7, activation="softmax")(x)  

    model = Model(vgg.input, x)
    
    return model

input_shape = (256,384,3)
model = Vgg16_GP_feature(input_shape)
model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])
model.summary()

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // 8,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 8,
    epochs = 20)

model.save('./vgg16_GP.hdf5')






