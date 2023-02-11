from utils.config import get_config
import urllib.request
import os
from utils.data import show_progress
import zipfile
from arguments import get_args
from utils.data import split_data
import tensorflow as tf

config = get_config('config.ini')

if not os.path.exists(config['DEFAULT']['DownloadDir']):
    os.makedirs(config['DEFAULT']['DownloadDir'])

print('Downloading dataset...')
urllib.request.urlretrieve(config['DEFAULT']['DataUrl'], 
                           config['DEFAULT']['DataFileName'],
                           reporthook=show_progress)
zip_ref = zipfile.ZipFile(config['DEFAULT']['DataFileName'], 'r')
zip_ref.extractall(config['DEFAULT']['DownloadDir'])
zip_ref.close()
os.remove(config['DEFAULT']['DataFileName'])

print('Dataset downloaded and extracted successfully')
print("Number of cat images:",len(os.listdir(config['DEFAULT']['CatImageDir'])))
print("Number of dog images:",len(os.listdir(config['DEFAULT']['DogImageDir'])))

try:
    os.makedirs(config['DEFAULT']["TrainingDir"])
    os.makedirs(config['DEFAULT']["TestingDir"])
    os.makedirs(config['DEFAULT']["TrainingCatsDir"])
    os.makedirs(config['DEFAULT']["TrainingDogsDir"])
    os.makedirs(config['DEFAULT']["TestingCatsDir"])
    os.makedirs(config['DEFAULT']["TestingDogsDir"])
except OSError:
    pass

print('Splitting the dataset into training and testing...')
args = get_args()
split_size = args.split_size
split_data(config['DEFAULT']['CatImageDir'], 
           config['DEFAULT']['TrainingCatsDir'], 
           config['DEFAULT']['TestingCatsDir'], 
           split_size)
split_data(config['DEFAULT']['DogImageDir'],
           config['DEFAULT']['TrainingDogsDir'],
           config['DEFAULT']['TestingDogsDir'],
           split_size)

print('Dataset split into training and testing successfully')
print("Number of cat training images:",len(os.listdir(config['DEFAULT']['TrainingCatsDir'])))
print("Number of dog training images:",len(os.listdir(config['DEFAULT']['TrainingDogsDir'])))
print("Number of cat testing images:",len(os.listdir(config['DEFAULT']['TestingCatsDir'])))
print("Number of dog testing images:",len(os.listdir(config['DEFAULT']['TestingDogsDir'])))

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
                                                    bath_size=args.batch_size,
                                                    class_mode='binary',
                                                    target_size=IMAGE_SIZE)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=args.aug_rescale)
test_generator = test_datagen.flow_from_directory(config['DEFAULT']['TestingDir'],
                                                  bath_size=args.batch_size,
                                                  class_mode='binary',
                                                  target_size=IMAGE_SIZE)
print('Training and testing augmented successfully')