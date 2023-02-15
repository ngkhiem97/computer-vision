import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
    image = tf.reshape(image, (28,28,1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32)/255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])
  
def get_training_dataset(batch_size, strategy):
      with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(-1)
      return dataset

def get_validation_dataset(strategy):
	with strategy.scope():
		dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
		dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
		dataset = dataset.batch(10000, drop_remainder=True)
		dataset = dataset.repeat()
	return dataset

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
	batch_train_ds = training_dataset.unbatch().batch(N)
	if tf.executing_eagerly():
		for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
			validation_digits = validation_digits.numpy()
			validation_labels = validation_labels.numpy()
			validation_bboxes = validation_bboxes.numpy()
			break
		for training_digits, (training_labels, training_bboxes) in batch_train_ds:
			training_digits = training_digits.numpy()
			training_labels = training_labels.numpy()
			training_bboxes = training_bboxes.numpy()
			break
	validation_labels = np.argmax(validation_labels, axis=1)
	training_labels = np.argmax(training_labels, axis=1)
	return (training_digits, training_labels, training_bboxes,
			validation_digits, validation_labels, validation_bboxes)