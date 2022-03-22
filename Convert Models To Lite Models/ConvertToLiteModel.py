import tensorflow as tf
import os

os.chdir("Convert Models To Lite Models")
SAVED_MODEL_DIRECTORY = "0 - Acceleration + Rot_Velocity + Magnetism\\model_E45-VL0.003844" # Example path

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIRECTORY)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Reduces both latency and file size (dynamic range quantization)
tflite_model = converter.convert()

# Save the lite model to this directory
with open('{}.tflite'.format(os.path.split(SAVED_MODEL_DIRECTORY)[1]), 'wb') as f:
  f.write(tflite_model)