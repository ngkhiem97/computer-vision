import tensorflow as tf

def feature_extractor(inputs):
  feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                            include_top=False,
                                                            weights='imagenet')(inputs)
  return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def get_model(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
  resnet_feature_extractor = feature_extractor(resize)
  resnet_classifier = classifier(resnet_feature_extractor)
  model = tf.keras.Model(inputs=inputs, outputs=resnet_classifier)
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model