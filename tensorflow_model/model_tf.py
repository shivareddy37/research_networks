
import tensorflow as tf

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 2

"""Model function for CNN."""
def model_promo(image_batch, mode):
    
    # Input Layer
    input_layer = tf.reshape(image_batch, [-1, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, 3], name = 'input')
 
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu, name = 'conv1')
 
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding = 'same', name = 'max_pool_1' )
 
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu, name = 'conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding = 'same', name = 'max_pool_2')

     # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu, name = 'conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding = 'same', name = 'max_pool_3')

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu, name = 'conv4')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding = 'same', name = 'max_pool_4')

    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu, name = 'conv5')
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, padding = 'same', name = 'max_pool_5')
 
    # Dense Layers
    pool5_flat = tf.reshape(pool5, [-1, 7* 7 * 256])
    dense1 = tf.layers.dense(inputs=pool5_flat, units= 2048, activation=tf.nn.relu, name = 'dense1')
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units= 256, activation=tf.nn.relu, name = 'dense2')
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense3 = tf.layers.dense(inputs=dropout2, units= 64, activation=tf.nn.relu, name = 'dense3')
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
   
 
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=2, name = 'logits') 
    output = tf.nn.softmax(logits, name='output')
    

    return output



