import urllib
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

INPUT_SHAPE = (150, 150, 3)

def get_inception_v3_model(weights_url, weights_file):
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = InceptionV3(input_shape=INPUT_SHAPE,
                                    include_top=False,
                                    weights=None)
    pre_trained_model.load_weights(weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(pre_trained_model.input, x)
    return model
