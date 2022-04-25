import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
def conv2ds_pool(inputs,channels,kernel,strides,activation,padding,crop_size):
    '''
    example: conv2ds(inputs,channels,kernel=3,strides=1,activation='relu',padding='same')
    '''
    conv1 = Conv2D(filters=channels, kernel_size=kernel, strides=strides,activation = activation, 
                   padding = padding, kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(filters=channels, kernel_size=kernel, strides=strides,activation = activation, 
                   padding = padding, kernel_initializer = 'he_normal')(conv1)
    copy_crop1 = CenterCrop(crop_size[0], crop_size[1])(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    return pool1,copy_crop1,conv1

def deconv2ds_pool(copy_crop,conv2d,channels,kernel,strides,activation,padding):
    '''
    example: conv2ds(inputs,channels,kernel=3,strides=1,activation='relu',padding='same')
    '''
    drop_prob = 0.5
    conv0_up = Conv2DTranspose(filters=int(channels*(1-drop_prob)), kernel_size=2, strides=2,
                   activation = activation, output_padding=None,
                   padding = 'valid')(conv2d)
    concate1 = concatenate([copy_crop,conv0_up], axis = 3)
    
    drop1 = concate1#Dropout(drop_prob)(concate1)
    conv1 = Conv2D(filters=int(channels*(1-drop_prob)), kernel_size=kernel, strides=strides,
                   activation = activation, 
                   padding = padding, kernel_initializer = 'he_normal')(drop1)
    conv1 = Conv2D(filters=int(channels*(1-drop_prob)), kernel_size=kernel, strides=strides,
                   activation = activation, 
                   padding = padding, kernel_initializer = 'he_normal')(conv1)
    pool1 = conv1#MaxPooling2D(pool_size=(2, 2))(conv1)
    return pool1
# Model
def UNet2d(pretrained_weights=None,input_size=(388,388,1)):
    '''
    example: UNet2d(pretrained_weights = None,input_size = (388,388,1))
    '''
    inputs = Input(input_size)
    img_padded = tf.pad(tensor=inputs, paddings=[[0, 0],[92,92],[92,92],[0,0]],
                        mode="SYMMETRIC",name='Padding')
    #-----------------------------------------------------------
    conv2ds_l1,copy_crop_l1,conv_l1 = conv2ds_pool(inputs=img_padded ,channels=64,kernel=3,strides=1,activation='relu',
                                           padding='valid',crop_size=[392,392])
    conv2ds_l2,copy_crop_l2,conv_l2 = conv2ds_pool(inputs=conv2ds_l1,channels=128,kernel=3,strides=1,activation='relu',
                                           padding='valid',crop_size=[200,200])
    conv2ds_l3,copy_crop_l3,conv_l3 = conv2ds_pool(inputs=conv2ds_l2,channels=256,kernel=3,strides=1,activation='relu',
                                           padding='valid',crop_size=[104,104])
    conv2ds_l4,copy_crop_l4,conv_l4 = conv2ds_pool(inputs=conv2ds_l3,channels=512,kernel=3,strides=1,activation='relu',
                                           padding='valid',crop_size=[56,56])
    conv2ds_l5,copy_crop_l5,conv_l5 = conv2ds_pool(inputs=conv2ds_l4,channels=1024,kernel=3,strides=1,activation='relu',
                                           padding='valid',crop_size=[28,28])
    #------------------------------------------------------------
    deconv2ds_l6 = deconv2ds_pool(copy_crop=copy_crop_l4,conv2d=conv_l5,channels=1024,
                                  kernel=3,strides=1,activation='relu',padding='valid')
    deconv2ds_l7 = deconv2ds_pool(copy_crop=copy_crop_l3,conv2d=deconv2ds_l6,channels=512,
                                  kernel=3,strides=1,activation='relu',padding='valid')
    deconv2ds_l8 = deconv2ds_pool(copy_crop=copy_crop_l2,conv2d=deconv2ds_l7,channels=256,
                                  kernel=3,strides=1,activation='relu',padding='valid')
    deconv2ds_l9 = deconv2ds_pool(copy_crop=copy_crop_l1,conv2d=deconv2ds_l8,channels=128,
                                  kernel=3,strides=1,activation='relu',padding='valid')
    conv10 = Conv2D(filters=1, kernel_size=1, strides=1,activation = 'sigmoid', 
                       padding = 'valid')(deconv2ds_l9)
    #------------------------------------------------------------
    model = Model(inputs = inputs, outputs = conv10,name='UNet')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model