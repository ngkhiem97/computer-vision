
import tensorflow as tf
from utils.data import preprocess_image_input
from model.resnet50 import get_model
from utils.config import get_config
from arguments import get_args

config = get_config('config.ini')
args = get_args()

(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)
model = get_model(optimizer=args.optimizer, loss=args.loss, metrics=args.metrics)
model.fit(train_X, training_labels, epochs=args.epochs, validation_data = (valid_X, validation_labels), batch_size=args.batch_size)
model.save(config['DEFAULT']['ModelDir'] + config['DEFAULT']['ModelName'])