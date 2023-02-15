from arguments import get_args
from utils.config import get_config
from model.cnn import compile_model
import tensorflow as tf
from utils.data import get_training_dataset, get_validation_dataset, dataset_to_numpy_util
import os
import datetime
import pickle as pkl

args = get_args()
config = get_config('config.ini')

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy()
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy()
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

global_batch_size = args.batch_size * strategy.num_replicas_in_sync

with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
    model = compile_model(inputs)
training_dataset = get_training_dataset(global_batch_size, strategy)
validation_dataset = get_validation_dataset(strategy)

history = model.fit(training_dataset, 
                    epochs=args.epochs, 
                    validation_data=validation_dataset, 
                    steps_per_epoch=args.steps_per_epoch//global_batch_size, 
                    validation_steps=args.validation_steps)
    
loss, classification_loss, bounding_box_loss, classification_accuracy, bounding_box_mse = model.evaluate(validation_dataset, steps=args.validation_steps)
print("Classification Accuracy: ", classification_accuracy)
print("Bounding Box MSE: ", bounding_box_mse)

os.makedirs(config['DEFAULT']['LogDir'], exist_ok=True)
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save(os.path.join(config['DEFAULT']['ModelDir'], 'model_' + dt + '.h5'))
with open(os.path.join(config['DEFAULT']['LogDir'], 'history_' + dt + '.pkl'), 'wb') as f:
    pkl.dump(history.history, f)
