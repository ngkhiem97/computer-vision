
import tensorflow as tf
from utils.data import preprocess_image_input
from model.resnet50 import define_compile_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)
model = define_compile_model()
EPOCHS = 4
history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data = (valid_X, validation_labels), batch_size=64)