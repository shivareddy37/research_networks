
from keras.models import Sequential
from keras.layers import Convolution2D,Cropping2D,MaxPooling2D
from keras.layers.core import  Dropout, Lambda
from keras.layers.core import Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.core import Activation




class Promotag_Classifier:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)

		opt = Adam(lr=10e-5, decay=10e-5 / 25)
		if K.image_data_format() == 'channels_first':
			inputShape = (depth,height,width)

		model.add(Convolution2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Convolution2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Convolution2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dropout(0.25))
		model.add(Dense(100))
		model.add(Activation("relu"))
		model.add(Dropout(0.25))

 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model


