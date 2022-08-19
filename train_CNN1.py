from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model



train_path = './finaldataset/train'
valid_path = './finaldataset/val'

train_batches = ImageDataGenerator(rotation_range=20, horizontal_flip=True, brightness_range=[0.2,1.0], preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=64)

valid_batches = ImageDataGenerator(rotation_range=20, horizontal_flip=True, brightness_range=[0.2,1.0], preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=64)


mobile = tf.keras.applications.mobilenet.MobileNet()

mobile.summary()

x = Flatten() (mobile.layers[-6].output)
output = Dense(units=28, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=output)
for layer in model.layers[:-23]:
    layer.trainable = False
#print('Number of layers in the base model: ',len(model.layers))
model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("model_sign_language.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=valid_batches, validation_steps=len(valid_batches), epochs=20, callbacks=[checkpoint])
model.save("model_sign_language.h5")
