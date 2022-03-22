import tensorflow as tf

SAVED_MODEL_DIRECTORY = ""

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIRECTORY)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Reduces both latency and file size (dynamic range quantization)
tflite_model = converter.convert()

# Save the lite model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)