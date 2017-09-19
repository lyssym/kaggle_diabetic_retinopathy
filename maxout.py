# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import numpy as np
from time import gmtime, strftime
from keras.layers import Input
from keras.layers import Conv2D, Dropout, Lambda
from keras.layers.pooling import MaxPool2D
from keras.layers import Dense, MaxoutDense, Flatten
from keras.initializers import Constant, Orthogonal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation
from keras.models import Model
from keras import optimizers
from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback

from process1 import generate_data
from config import batch_size, num_chunk

leakiness = 0.5
epochs = 30
num_channels = 3
input_width = 512
input_height = 512
output_dim = 512
momentum = 0.90
lr_scale = 4.00
learing_rate = 0.0020
decay = 1e-6

LEARNING_RATE_SCHEDULE = {
    1: 0.0020 * lr_scale,
    num_chunk // 10: 0.0015 * lr_scale,
    num_chunk * 2 // 10: 0.0010 * lr_scale,
    num_chunk * 4 // 10: 0.00075 * lr_scale,
    num_chunk * 5 // 10: 0.00050 * lr_scale,
    num_chunk * 6 // 10: 0.00010 * lr_scale,
    num_chunk * 7 // 10: 0.00001 * lr_scale,
    num_chunk * 9 // 10: 0.000001 * lr_scale,
}


def reshape_batch(x, batch_size):
    row, col = x.shape
    dst_tensor = K.reshape(x, (batch_size, int(row * col // batch_size)))
    return dst_tensor


def build_model():
    input_1 = Input(batch_shape=(batch_size, num_channels, input_width, input_height))
    conv_1_1 = Conv2D(32,
                      kernel_size=(7, 7),
                      strides=(2, 2),
                      padding='same',
                      use_bias=True,
                      activation='linear',
                      data_format='channels_first',
                      kernel_initializer=Orthogonal(1.0),
                      bias_initializer=Constant(0.1),
                      name='conv1_1')(input_1)

    relu_1_1 = LeakyReLU(leakiness)(conv_1_1)
    pool_1_1 = MaxPool2D((3, 3),
                         strides=(2, 2),
                         data_format='channels_first',
                         batch_size=batch_size)(relu_1_1)

    conv_2_1 = Conv2D(32, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv2_1')(pool_1_1)
    relu_2_1 = LeakyReLU(leakiness)(conv_2_1)
    conv_2_2 = Conv2D(32, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv2_2')(relu_2_1)
    relu_2_2 = LeakyReLU(leakiness)(conv_2_2)
    pool_2_2 = MaxPool2D((3, 3),
                         strides=(2, 2),
                         data_format='channels_first',
                         batch_size=batch_size)(relu_2_2)

    conv_3_1 = Conv2D(64, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv3_1')(pool_2_2)
    relu_3_1 = LeakyReLU(leakiness)(conv_3_1)
    conv_3_2 = Conv2D(64, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv3_2')(relu_3_1)
    relu_3_2 = LeakyReLU(leakiness)(conv_3_2)
    pool_3_2 = MaxPool2D((3, 3),
                         strides=(2, 2),
                         data_format='channels_first',
                         batch_size=batch_size)(relu_3_2)

    conv_4_1 = Conv2D(128, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv4_1')(pool_3_2)
    relu_4_1 = LeakyReLU(leakiness)(conv_4_1)
    conv_4_2 = Conv2D(128, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv4_2')(relu_4_1)
    relu_4_2 = LeakyReLU(leakiness)(conv_4_2)
    conv_4_3 = Conv2D(128, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv4_3')(relu_4_2)
    relu_4_3 = LeakyReLU(leakiness)(conv_4_3)
    conv_4_4 = Conv2D(128, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv4_4')(relu_4_3)
    relu_4_4 = LeakyReLU(leakiness)(conv_4_4)
    pool_4_4 = MaxPool2D((3, 3),
                         strides=(2, 2),
                         data_format='channels_first',
                         batch_size=batch_size)(relu_4_4)

    conv_5_1 = Conv2D(256, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv5_1')(pool_4_4)
    relu_5_1 = LeakyReLU(leakiness)(conv_5_1)
    conv_5_2 = Conv2D(256, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv5_2')(relu_5_1)
    relu_5_2 = LeakyReLU(leakiness)(conv_5_2)
    conv_5_3 = Conv2D(256, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv5_3')(relu_5_2)
    relu_5_3 = LeakyReLU(leakiness)(conv_5_3)
    conv_5_4 = Conv2D(256, (3, 3),
                     padding='same',
                     use_bias=True,
                     activation='linear',
                     data_format='channels_first',
                     kernel_initializer=Orthogonal(1.0),
                     bias_initializer=Constant(0.1),
                     name='conv5_4')(relu_5_3)
    relu_5_4 = LeakyReLU(leakiness)(conv_5_4)
    pool_5_4 = MaxPool2D((3, 3),
                         strides=(2, 2),
                         data_format='channels_first',
                         batch_size=batch_size,
                         name='coarse_last_pool')(relu_5_4)

    dropout_1 = Dropout(0.5)(pool_5_4)
    flatten_1 = Flatten()(dropout_1)
    maxout_dense_0 = MaxoutDense(output_dim=512,
                                 dtype='float32',
                                 init='orthogonal',
                                 name='first_fc_0')(flatten_1)

    new_batch = batch_size // 2
    reshape_tensor = Lambda(reshape_batch, arguments={'batch_size': new_batch})(maxout_dense_0)

    dropout_2 = Dropout(0.5)(reshape_tensor)
    maxout_dense_1 = MaxoutDense(output_dim=512,
                                 init='orthogonal',
                                 name='first_fc_1')(dropout_2)

    dropout_3 = Dropout(0.5)(maxout_dense_1)
    dense_1 = Dense(4,
                    kernel_initializer=Orthogonal(1.0),
                    bias_initializer=Constant(0.1),
                    name='last_dense')(dropout_3)

    reshape_tensor = Lambda(reshape_batch, arguments={'batch_size': batch_size})(dense_1)
    predictions = Activation('softmax', name='last_out')(reshape_tensor)

    model = Model(inputs=input_1, outputs=predictions)
    return model


def optime_model(model, learing_rate=learing_rate, decay=decay,
                 momentum=0.9, nesterov=True, clipnorm=10):
    sgd = optimizers.SGD(lr=learing_rate,
                         decay=decay,
                         momentum=momentum,
                         nesterov=nesterov,
                         clipnorm=clipnorm)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def save_weight(model, weight_name='model/weight.h5'):
    model.save_weights(weight_name)


def save_model(model, network_name='model/network.json'):
    json_string = model.to_json()
    with open(network_name, 'w') as f:
        f.write(json_string)
        f.close()


def load_model(network_name='model/network.json'):
    model = model_from_json(open(network_name, encoding='utf-8').read())
    return model


def load_weight(model, weight_name='model/weight.h5'):
    model.load_weight(weight_name)
    return model


class Decay_lr(Callback):
    def __init__(self, epoch, batch):
        super(Decay_lr, self).__init__()
        self.n_epoch = epoch
        self.n_batch = batch

    def on_epoch_begin(self, epoch, logs=None):
        old_lr = learing_rate * (1 - 3*epoch /(epochs*epochs))
        if epoch > 0 and epoch % self.n_epoch == 0:
            new_lr = old_lr * 0.988
        else:
            new_lr = old_lr * 0.999
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_batch_begin(self, batch, logs=None):
        old_lr = K.get_value(self.model.optimizer.lr)
        if batch in LEARNING_RATE_SCHEDULE:
            new_lr = LEARNING_RATE_SCHEDULE.get(batch)
        elif batch > 0 and batch % self.n_batch == 0:
            new_lr = old_lr * 0.99999
        else:
            new_lr = old_lr
        K.set_value(self.model.optimizer.lr, new_lr)


def train_model(model, generator):
    model_id = strftime("%Y_%m_%d_%H%M%S", gmtime())
    save_model_name = 'model/best_model_' + model_id + '.h5'
    save_best = ModelCheckpoint(save_model_name,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)
    decay_lr = Decay_lr(5, num_chunk//100)
    result = model.fit_generator(generator,
                        steps_per_epoch=num_chunk,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[save_best, decay_lr],
                        max_queue_size=4,
                        workers=4,
                        use_multiprocessing=True)
    return result


if __name__ == '__main__':
    # prefix_train = 'train_raw/'
    # prefix_test = 'test_raw/'
    # generator = generate_data(prefix_train, prefix_test,
    #                           image_width=input_width,
    #                           image_height=input_height,
    #                           num_channels=3,
    #                           num_chunk=num_chunk,
    #                           rng=np.random,
    #                           chunk_size=batch_size)
    model = build_model()
    model = optime_model(model)
    print(model.input)
    # train_model(model, generator)
    save_model(model)
    # save_weight(model)
