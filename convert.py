import tensorflow as tf

model = tf.keras.models.load_model('model_sign_language.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)