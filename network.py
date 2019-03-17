import keras as k
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, concatenate
from keras.models import Model

import config


def backbone(x):
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    encode = GlobalAveragePooling2D()(x)

    return encode


def siamese_net(input_a, input_b):
    inp = Input(config.INPUT_SHAPE)
    encode = backbone(inp)
    backbone_model = Model(inp, encode)

    encode_a = backbone_model(input_a)
    encode_b = backbone_model(input_b)

    concatenated = concatenate([encode_a, encode_b])
    x = Dense(128)(concatenated)
    x = Dense(64)(x)
    predict = Dense(1)(x)
    return predict


def build_model():
    input_a = Input(config.INPUT_SHAPE)
    input_b = Input(config.INPUT_SHAPE)
    predict = siamese_net(input_a, input_b)
    model = Model([input_a, input_b], predict)

    return model

model = build_model()
model.summary()