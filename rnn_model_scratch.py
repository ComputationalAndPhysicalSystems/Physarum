import numpy as np


from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense,Flatten,Input,concatenate,Dot, Conv2D,Reshape, MaxPooling2D, UpSampling2D,Conv3DTranspose, ZeroPadding2D,Conv3D,Conv2DTranspose, BatchNormalization, Dropout
from keras.models import Model,Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import TensorBoard
import numpy as np





import sys
from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.engine import InputSpec
import numpy as np
droprate=0.4

import tensorflow as tf
droprate=0.4
#############################INITATE LAYERS
import tensorflow as tf
from keras.layers import Lambda


from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.engine import InputSpec
import numpy as np
droprate=0.4

import tensorflow as tf
droprate=0.4
#############################INITATE LAYERS
import tensorflow as tf
from keras.layers import Lambda

def SubpixelConv2D(input_shape, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape)
#########################DEFIEN LOSSES
def gdl_loss(y_pred, y_true):
    """
    Calculates the sum of GDL losses between the predicted and ground truth images.
    @param y_pred: The predicted CTs.
    @param y_true: The ground truth images
    @param alpha: The power to which each gradient term is raised.
    @param batch_size_tf batch size
    @return: The GDL loss.
    """
    # calculate the loss for each scale

    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(y_pred, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(y_pred, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(y_true, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(y_true, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    gdl=tf.reduce_sum((grad_diff_x ** 2 + grad_diff_y ** 2))/tf.cast(10,tf.float32)

    return gdl


HUBER_DELTA = 0.5


def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))

    return K.sum(x)


def GDL1(y_true, y_pred):
    """
    Calculates the sum of GDL losses between the predicted and ground truth images.
    @param y_pred: The predicted CTs.
    @param y_true: The ground truth images
    @param alpha: The power to which each gradient term is raised.
    @param batch_size_tf batch size
    @return: The GDL loss.
    """
    # calculate the loss for each scale

    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(y_pred, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(y_pred, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(y_true, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(y_true, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    gdl = tf.reduce_sum((grad_diff_x ** 2 + grad_diff_y ** 2)) / tf.cast(10, tf.float32)

    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    l1 = K.sum(x)
    gdl1 = (l1 / 100) + (gdl *  10000)

    return gdl1
##############################IMPORT DATA
import h5py

hf = h5py.File('data_slime_aug1st_equal20_2_3_tempo_split.h5', 'r', driver='core')
bigting = hf.get('train')
train = np.array(bigting)
bigting = hf.get('test')
test = np.array(bigting)
bigting = hf.get('test2')
test2 = np.array(bigting)
bigting = hf.get('train2')
train2 = np.array(bigting)
bigting = hf.get('tempo')
jumps = np.array(bigting)
hf.close()

train = train[456:,:,:,:,:]
test =   test[456:,:,:,:,:]
test2 = test2[456:,:,:,:,:]
train2 = train2[456:,:,:,:,:]
jumps = jumps[456:,]

#sdfsd
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))

    return K.sum(x)
##############################################################################BEGIN MODEL




seq = Sequential()

inp_layer = Input(name='the_input', dtype='float32',batch_shape=(5,2,64,64,1))
x1 = (Conv3D(filters=40,kernel_size=(4,4,4),strides=(2,1,1),kernel_initializer='lecun_uniform',padding='same',activation='tanh'))(inp_layer)
x1 = Dropout(.2)(x1)
inp_layer2 = Input(name='the_input2', dtype='float32',batch_shape=(5,2,64,64,1))
x2 = (Conv3D(filters=40,kernel_size=(4,4,4),strides=(2,1,1),kernel_initializer='lecun_uniform',padding='same',activation='tanh'))(inp_layer2)
x2 = Dropout(.2)(x2)
x = concatenate(inputs = [x1, x2], axis = 4)
x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(x)
x = (BatchNormalization())(x)
x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(x)
x = (BatchNormalization())(x)
x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4,recurrent_dropout=0.3)(x)
x = Conv3DTranspose(filters=40,kernel_size=(4,4,4),strides=(3,1,1),padding='same',activation='tanh')(x)

x = Dropout(.2)(x)

#x = (BatchNormalization())(x)

x1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(x)
#x = (BatchNormalization())(x)
x2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(x)
merged = concatenate(inputs = [x1, x2], axis = 4)
x1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(merged)
#x1 = (BatchNormalization())(x1)
x2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4,recurrent_dropout=0.3)(merged)
#x2 = (BatchNormalization())(x2)
merged = concatenate(inputs = [x1, x2], axis = 4)
tempout = (Conv3D(filters=40,kernel_size=(4,4,4),kernel_initializer='lecun_uniform',padding='same',activation='tanh'))(merged)


tempout = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_initializer='glorot_uniform', unit_forget_bias=True,
               dropout=0.4,recurrent_dropout=0.3)(tempout)
tempout = (BatchNormalization())(tempout)
tempout = (Flatten())(tempout)
tempout = (Dense(1, activation='sigmoid'))(tempout)
x1 = (Conv3D(filters=1,kernel_size=(11,11,1),kernel_initializer='random_uniform',padding='same',activation='tanh'))(merged)
x2 = (Conv3D(filters=1,kernel_size=(11,11,1),kernel_initializer='random_uniform',padding='same',activation='tanh'))(merged)

seq = Model(inputs=[inp_layer,inp_layer2], outputs=[x1,x2,tempout])
seq.summary()
seq.compile(optimizer='adam', loss="mse")

#mc = ModelCheckpoint('weights_mcd{epoch:08d}.h5', save_weights_only=True, period=3)

history = seq.fit([train,train2],
        epochs= 30,
        batch_size = 5,
        shuffle=True,
        y= [test,test2,jumps],
	validation_split=0.1
        #callbacks=[mc]
                 )

#X_pred = seq.predict([train,train2], batch_size = 5)

#np.save('X_pred2dlstm_braided', X_pred)

saveloss = np.array(history.history["loss"])

#np.save("loss_across_epochs2d_braided_lstm",saveloss)

seq.save('seq_20_23_pred_braided_model.h5')
