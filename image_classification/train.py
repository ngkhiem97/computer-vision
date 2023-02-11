import tensorflow as tf
from arguments import get_args
from utils.config import get_config
from model.inception_v3 import get_inception_v3_model
import os
import datetime
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

config = get_config('config.ini')
args = get_args()

print('Augmenting the training and testing...')
IMAGE_SIZE = (150, 150)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=args.aug_rescale,
                                                                rotation_range=args.aug_rotation,
                                                                width_shift_range=args.aug_width,
                                                                height_shift_range=args.aug_height,
                                                                shear_range=args.aug_shear,
                                                                zoom_range=args.aug_zoom,
                                                                horizontal_flip=args.aug_horiz,
                                                                vertical_flip=args.aug_vert,
                                                                fill_mode=args.aug_fill)
train_generator = train_datagen.flow_from_directory(config['DEFAULT']['TrainingDir'],
                                                    batch_size=args.batch_size,
                                                    class_mode='binary',
                                                    target_size=IMAGE_SIZE)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=args.aug_rescale)
test_generator = test_datagen.flow_from_directory(config['DEFAULT']['TestingDir'],
                                                  batch_size=args.batch_size,
                                                  class_mode='binary',
                                                  target_size=IMAGE_SIZE)
print('Training and testing augmented successfully')

print('Building the model...')
model_dir = config['DEFAULT']['ModelDir']
weights_url = config['DEFAULT']['ModelWeightsUrl']
weights_file = model_dir + config['DEFAULT']['ModelWeightsFile']
model = get_inception_v3_model(weights_url, weights_file)
print('Model created successfully')

print('Compiling the model...')
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate),
              loss=args.loss,
              metrics=[args.metrics]) 
print('Model compiled successfully')

print('Training the model...')
log_dir = config['DEFAULT']['LogDir'] + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
if not log_dir:
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs=args.epochs,
                    verbose=1,
                    callbacks=[tensorboard_callback])
print('Model trained successfully')

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()