import os
import cv2
import numpy as np
import concurrent.futures
import urllib.request
import zipfile
from tqdm import tqdm
from arguments import get_args
from utils.config import get_config
from utils.data import show_progress, split_data

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_extract_dataset(config):
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
    print("Number of cat images:", len(os.listdir(config['DEFAULT']['CatImageDir'])))
    print("Number of dog images:", len(os.listdir(config['DEFAULT']['DogImageDir'])))

def split_and_process_images(config):
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
    print("Number of cat training images:", len(os.listdir(config['DEFAULT']['TrainingCatsDir'])))
    print("Number of dog training images:", len(os.listdir(config['DEFAULT']['TrainingDogsDir'])))
    print("Number of cat testing images:", len(os.listdir(config['DEFAULT']['TestingCatsDir'])))
    print("Number of dog testing images:", len(os.listdir(config['DEFAULT']['TestingDogsDir'])))

def process_image(file, split, category, data_dir):
    path = os.path.join(data_dir, split, category, file)
    image = cv2.imread(path)
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)
    phase_spectrum = ((phase_spectrum + np.pi) / (2 * np.pi)) * 255
    new_path_mag = os.path.join(data_dir + '-fft/mag', split, category, file)
    new_path_phase = os.path.join(data_dir + '-fft/phase', split, category, file)
    cv2.imwrite(new_path_mag, magnitude_spectrum)
    cv2.imwrite(new_path_phase, phase_spectrum)

if __name__ == "__main__":
    config = get_config('config.ini')
    data_dir = config['DEFAULT']['DataDir']
    
    download_and_extract_dataset(config)
    split_and_process_images(config)

    for split in ['training', 'testing']:
        for category in ['cats', 'dogs']:
            create_dir(os.path.join(data_dir + '-fft/mag', split, category))
            create_dir(os.path.join(data_dir + '-fft/phase', split, category))
            
            files = os.listdir(os.path.join(data_dir, split, category))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_image, 
                             files, 
                             [split] * len(files), 
                             [category] * len(files), 
                             [data_dir] * len(files))
