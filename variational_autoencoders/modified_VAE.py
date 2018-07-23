from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model
from keras import backend as K
from keras import metrics

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import random
import cv2
import keras
import math
# %matplotlib inline

## to force the program to run on cpu uncomment the below two lines 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# network parameters


original_dim = 256*256
input_shape = (256,256,1 )
intermediate_dim = 512
latent_dim = 256



def get_kernals_output(model, layer_index, model_input, training_flag = True):
    get_outputs = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_index].output])
    kernals_output = get_outputs([model_input, training_flag])[0]
    return kernals_output


def combine_ouput_images(kernals_output):

    output_count = kernals_output.shape[1]
    width = int(math.sqrt(output_count))
    height = int(math.ceil(float(output_count)/width))

    if len(kernals_output.shape ) == 4:
        output_shape = kernals_output.shape[2:]
        image = np.zeros((height * output_shape[0], width * output_shape[1]), dtype = kernals_output.dtype)
        for index , output in enumerate(kernals_output[0]):
            i = int (index / width)
            j = index % width
            image[i * output_shape[0]: (i +1)* output_shape[0], j * output_shape[1]: (j+1)* output_shape[1]] = output

        return image


def get_model_layers_output_combined_image(model, model_input, training_flag = True):
    for layer_idx in range(len(model.layers)):
        kernals_output = get_kernals_output(model, layer_idx, model_input, training_flag)
        combine_ouput_images = combine_ouput_images(kernals_output)

        if combine_ouput_images is not None:
            print(model.layers[layer_idx].name)
            plt.matshow(combine_ouput_images.T, vmin = 0.0, vmax = 1.0)
            plt.show()


class OutputImages(keras.callbacks.Callback):
    def __init__(self, input_data):
        self.input_data = input_data

    def _on_epoch_end(self, epoch = 1, logs = {}):
        
        get_models_layers_output_combined_image(self.model, self.input_data)


def preprocess_image(image):
    image = image.astype('float32')
    image = image - image.min()
    image = image / image.max()
    image_size = image.shape[1]
    global original_dim 
    original_dim = image_size * image_size
    # image = np.reshape(image, [original_dim, -1])
    return image


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=25 ,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# # MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# image_size = x_train.shape[1]
# original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

def image_generator(fnames, train, validation_split=0.8,
                    batch_size=25, image_size=256
                    ):  # type: (List[str], bool, float, int) -> Iterator[np.ndarray]
    if train:
        filenames = fnames[0:int(len(fnames) * validation_split)]
    else:
        filenames = fnames[int(len(fnames) * validation_split):]

    random.shuffle(filenames)
    

    input_images = []
    output_images = []
    for ii, fname in enumerate(filenames):
        img = img_to_array(load_img(fname, target_size=(image_size,
            image_size)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape((image_size, image_size,1))
        # img = np.dstack((img, img, img))
        im = preprocess_image(img)

        input_images.append(im)
        output_images.append(im)
        if (ii + 1) % batch_size == 0:
            yield np.array(input_images), np.array(output_images)
            input_images = []
            output_images = []
        

            
        




## VAE model = encoder + decoder
## build encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# conv1 = Conv2D(16, (3,3), padding = 'same', activation = 'relu', name = 'conv1')(inputs)
# conv2 = Conv2D(16,(3,3), padding = 'same', activation = 'relu', name = 'conv2', strides = 2)(conv1)
# conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'conv3')(conv2)
# conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, name = 'conv4')(conv3)
# conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'conv5')(conv4)
# conv6 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=2, name = 'conv6')(conv5)
# conv7 = Conv2D(128, (5, 5), activation='relu', padding='same', name = 'conv7')(conv6)
# flat = Flatten()(conv7)
# fc1 = Dense(1024, activation = 'relu', name = 'fc1')(flat)
# drop = Dropout(0.4)(fc1)
# fc2 = Dense(intermediate_dim, activation='relu', name = 'fc2')(drop)
# z_mean = Dense(latent_dim, name='z_mean')(fc2)
# z_log_var = Dense(latent_dim, name='z_log_var')(fc2)

# # use reparameterization trick to push the sampling out as input
# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


# # instantiate encoder model
# #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# #encoder.summary()
# #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# # build decoder model
# #latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

# #d_fc1 = Dense(intermediate_dim, activation='relu', name = 'd_fc1')(latent_inputs)

# d_fc1 = Dense(intermediate_dim, activation='relu', name = 'd_fc1')(z)
# d_fc2 = Dense(1024, activation = 'relu', name = 'd_fc2')(d_fc1)
# d_reshape = Reshape((32,32,1),input_shape = (1024,) )(d_fc2)
# d_conv_t_1 = Conv2DTranspose(128, (5, 5), strides=1, activation='relu', padding='same')(d_reshape)
# d_conv_t_2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(d_conv_t_1)
# d_conv_t_3 = Conv2DTranspose(64, (3, 3), strides=1, activation='relu', padding='same')(d_conv_t_2)
# d_conv_t_4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(d_conv_t_3)
# d_conv_t_5 = Conv2DTranspose(32, (3, 3), strides=1, activation='relu', padding='same')(d_conv_t_4)
# d_conv_t_6 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(d_conv_t_5)
# outputs = Conv2D(1, (3, 3), activation='relu', padding='same')(d_conv_t_6)
# vgg_vae = Model (inputs, outputs)
# # instantiate decoder model

# #decoder = Model(latent_inputs, outputs, name='decoder')
# #decoder.summary()
# #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)





# instantiate VAE model
#outputs = decoder(encoder(inputs)[2])
#vae = Model(inputs, outputs, name='vae_mlp')
# vae = Model(inputs, outputs, name='vae_mlp')

img_input = Input(shape=(256,256,1), name = 'encoder_input')

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
# f1 = x
# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
# f2 = x

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
# f3 = x

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
# f4 = x

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
f5 = x

x = Flatten(name='flatten')(x)
# #x = Dense(4096, activation='relu', name='fc1')(x)
# #x = Dropout(0.5)(x)
# #x = Dense(4096, activation='relu', name='fc2')(x)
# #x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu', name = 'fc3')(x)


inter = Dense(512, activation='relu', name = 'inter')(x)
z_mean = Dense(21, name='z_mean')(inter)
z_log_var = Dense(21, name='z_log_var')(inter)

z = Lambda(sampling, output_shape=(21,), name='z')([z_mean, z_log_var])

o = Dense(512, activation='relu', name = 'd_fc1')(z)
o = Dense(1024, activation = 'relu', name = 'd_fc2')(o)
o = Dropout(0.5)(o)
# #o = Dense(4096, activation='relu', name='d_fc3')(o)
# #o = Dropout(0.5)(o)
# #o = Dense(4096, activation='relu', name='d_fc4')(o)
o = Reshape((8,8,16),input_shape = (1024,) )(o)

o = UpSampling2D(size = (2,2))(o)
o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv1' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv2' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block5_conv3' )(o)
# d_f5 = o

o = UpSampling2D(size = (2,2))(o)
o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv1' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv2' )(o)
# o = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='d_block4_conv3' )(o)
# d_f4 = o

o = UpSampling2D(size = (2,2))(o)
o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv1' )(o)
# o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv2' )(o)
# o = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='d_block3_conv3' )(o)
# d_f3 = o

o = UpSampling2D(size = (2,2))(o)
o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='d_block2_conv1' )(o)
# o = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='d_block2_conv2' )(o)
# d_f2 = o

o = UpSampling2D(size = (2,2))(o)
o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='d_block1_conv1' )(o)
# o = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='d_block1_conv2' )(o)
# d_f1 = o

img_output = Conv2D(1, (3, 3), activation='relu', padding='same')(o)

vgg_vae = Model(img_input,img_output)


def my_vae_loss(y_true, y_pred):
    xent_loss = 256 * 256 * metrics.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
   
    #models = (encoder, decoder)
    fnames = glob('/home/shiva/data/m_turk_products/*.jpg')[0:600]
    # VAE loss = mse_loss or xent_loss + kl_loss
    # if args.mse:
    #     reconstruction_loss = mse(img_input, img_output)
    # else:
    #     reconstruction_loss = binary_crossentropy(img_input,
    #                                               img_input)

    #reconstruction_loss *= original_dim
    #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    #kl_loss = K.sum(kl_loss, axis=-1)
    #kl_loss *= -0.5
    #vae_loss = K.mean(reconstruction_loss + kl_loss)
    #vae.add_loss(vae_loss)
    vgg_vae.compile(optimizer='adam', loss= 'mse', metrics=['accuracy'])
    vgg_vae.summary()
    plot_model(vgg_vae,to_file='vgg_vae.png',show_shapes=True)

    bs = 20

    for i in range(1000):
        if args.weights:
            vgg_vae = vgg_vae.load_weights(args.weights)
        else:
            generator = image_generator(fnames, train=True, image_size=256, batch_size=bs)
            # train the autoencoder
            # import ipdb; ipdb.set_trace()
            print ('Iteration : ' , i)
            output_images = OutputImages(generator)
            result = vgg_vae.fit_generator(generator,
                steps_per_epoch=15,
                epochs=1,
                validation_data=image_generator(fnames, train=False,image_size=256, batch_size=bs),
                validation_steps = 1,
                verbose=1, callbacks = [output_images])
            im = cv2.imread("/home/shiva/data/m_turk_products/113.jpg")[:,:,0:1]
            im = preprocess_image(im)
            im = cv2.resize(im, (256,256))
            im = im.reshape((256, 256,1))
            im = np.expand_dims(im, 0)
            im_1 = vgg_vae.predict(im)[0, :, :, 0]
            cv2.imwrite("/home/shiva/img_vgg.jpg", (im_1/im_1.max()*255).astype('uint8'))
            
            print("Test-loss:", np.mean(result.history["val_loss"]))
            


            vgg_vae.save_weights('vgg_vae.h5')
    # plt.plot(result.history['loss'])
    # plt.plot(result.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")
    
