import urllib
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import layers
from keras import Model

INPUT_SHAPE = (150, 150, 3)

def get_mobile_net_v2_model(weights_url, weights_file):
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = MobileNetV2(input_shape=INPUT_SHAPE,
                                                          include_top=False,
                                                          weights=None)
    pre_trained_model.load_weights(weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('out_relu')
    last_output = last_layer.output
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(pre_trained_model.input, x)
    return model