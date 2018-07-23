

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, Reshape,UpSampling2D, Conv2DTranspose, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model
from keras import backend as K
from keras import metrics
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import random
import cv2
import tensoflow as tf


base_model = InceptionV3(weights='imagenet', include_top=True)

base_model.summary()


# original_dim = 256*256
# input_shape = (256,256,1 )
# intermediate_dim = 512
# latent_dim = 32
# # from glob import glob
# # import shutil
# # import os, sys
# # import cv2

# # # train_path = "/home/shiva/data/m_turk_products/train/"
# # # val_path = "/home/shiva/data/m_turk_products/val/"


# # # if not os.path.exists(train_path):
# # # 	os.makedirs(train_path)
# # # if not os.path.exists(val_path):
# # # 	os.makedirs(val_path)
# # # name = []
# # # for filename in glob("/home/shiva/data/m_turk_products/*.jpg"):
# # # 	name.append(filename)

# # # for idx, fname in enumerate(name):
# # # 	# print (fname)
# # # 	if idx < 62552:
# # # 		shutil.move(fname, train_path)
# # # 	else:
# # # 		shutil.move(fname, val_path)
# # for filename in glob('/home/shiva/data/m_turk_products/*.jpg'):
# # 	img = cv2.imread(filename)
# # 	# resized_image = cv2.resize(img, (256,256))
# # 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 	cv2.imwrite(filename, img)


# def get_model_memory_usage(batch_size, model):
#     import numpy as np
#     from keras import backend as K

#     shapes_mem_count = 0
#     for l in model.layers:
#         single_layer_mem = 1
#         for s in l.output_shape:
#             if s is None:
#                 continue
#             single_layer_mem *= s
#         shapes_mem_count += single_layer_mem

#     trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
#     non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

#     total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
#     gbytes = np.round(total_memory / (1024.0 ** 3), 3)
#     return gbytes
# def sampling(args):
#     """Reparameterization trick by sampling fr an isotropic unit Gaussian.
#     # Arguments:
#         args (tensor): mean and log of variance of Q(z|X)
#     # Returns:
#         z (tensor): sampled latent vector
#     """

#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     # by default, random_normal has mean=0 and std=1.0
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon

 

# img_input = Input(shape=(256,256,1), name = 'encoder_input')

# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
# # f1 = x
# # Block 2
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
# # f2 = x

# # Block 3
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
# # f3 = x

# # Block 4
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
# # f4 = x

# # Block 5
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
# # f5 = x

# x = Flatten(name='flatten')(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation = 'relu', name = 'fc3')(x)


# inter = Dense(512, activation='relu', name = 'inter')(x)
# z_mean = Dense(32, name='z_mean')(inter)
# z_log_var = Dense(32, name='z_log_var')(inter)

# z = Lambda(sampling, output_shape=(32,), name='z')([z_mean, z_log_var])

# o = Dense(512, activation='relu', name = 'd_fc1')(z)
# o = Dense(1024, activation = 'relu', name = 'd_fc2')(o)
# o = Dropout(0.5)(o)
# o = Dense(4096, activation='relu', name='d_fc3')(o)
# o = Dropout(0.5)(o)
# o = Dense(4096, activation='relu', name='d_fc4')(o)
# o = Reshape((8,8,64),input_shape = (4096,) )(o)

# o = UpSampling2D(size = (2,2))(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv1' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv2' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv3' )(o)
# # d_f5 = o

# o = UpSampling2D(size = (2,2))(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv1' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv2' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv3' )(o)
# # d_f4 = o

# o = UpSampling2D(size = (2,2))(o)
# o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv1' )(o)
# o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv2' )(o)
# o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv3' )(o)
# # d_f3 = o

# o = UpSampling2D(size = (2,2))(o)
# o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='d_block2_conv1' )(o)
# o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='d_block2_conv2' )(o)
# # d_f2 = o

# o = UpSampling2D(size = (2,2))(o)
# o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='d_block1_conv1' )(o)
# o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='d_block1_conv2' )(o)
# # d_f1 = o

# img_output = Conv2D(1, (3, 3), activation='relu', padding='same')(o)
# vgg_vae = Model(img_input,img_output)



# # # VAE model = encoder + decoder
# # # build encoder model
# # inputs = Input(shape=input_shape, name='encoder_input')
# # conv1 = Conv2D(16, (3,3), padding = 'same', activation = 'relu', name = 'conv1')(inputs)
# # conv2 = Conv2D(16,(3,3), padding = 'same', activation = 'relu', name = 'conv2', strides = 2)(conv1)
# # conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv3')(conv2)
# # conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, name = 'conv4')(conv3)
# # conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv5')(conv4)
# # conv6 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=2, name = 'conv6')(conv5)
# # conv7 = Conv2D(128, (5, 5), activation='relu', padding='same', name = 'conv7')(conv6)
# # flat = Flatten()(conv7)
# # fc1 = Dense(1024, activation = 'relu', name = 'fc1')(flat)
# # drop = Dropout(0.4)(fc1)
# # fc2 = Dense(intermediate_dim, activation='relu', name = 'fc2')(drop)
# # z_mean = Dense(latent_dim, name='z_mean')(fc2)
# # z_log_var = Dense(latent_dim, name='z_log_var')(fc2)

# # # use reparameterization trick to push the sampling out as input
# # # note that "output_shape" isn't necessary with the TensorFlow backend
# # z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


# # # instantiate encoder model
# # #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# # #encoder.summary()
# # #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# # # build decoder model
# # #latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

# # #d_fc1 = Dense(intermediate_dim, activation='relu', name = 'd_fc1')(latent_inputs)

# # d_fc1 = Dense(intermediate_dim, activation='relu', name = 'd_fc1')(z)
# # d_fc2 = Dense(1024, activation = 'relu', name = 'd_fc2')(d_fc1)
# # d_reshape = Reshape((32,32,1),input_shape = (1024,) )(d_fc2)
# # d_conv_t_1 = Conv2DTranspose(128, (5, 5), strides=1, activation='relu', padding='same')(d_reshape)
# # d_conv_t_2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(d_conv_t_1)
# # d_conv_t_3 = Conv2DTranspose(64, (3, 3), strides=1, activation='relu', padding='same')(d_conv_t_2)
# # d_conv_t_4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(d_conv_t_3)
# # d_conv_t_5 = Conv2DTranspose(32, (3, 3), strides=1, activation='relu', padding='same')(d_conv_t_4)
# # d_conv_t_6 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(d_conv_t_5)
# # outputs = Conv2D(1, (3, 3), activation='relu', padding='same')(d_conv_t_6)
# # vgg_vae = Model(inputs,outputs)

# # vgg_vae.summary()

# # plot_model(vgg_vae,to_file='vgg_vae.png',show_shapes=True)

# print(get_model_memory_usage(25,vgg_vae))
