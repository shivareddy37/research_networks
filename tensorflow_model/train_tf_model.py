import tensorflow as tf
from model_tf import model_promo 


# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
IMAGE_SIZE = 224
IMAGE_SHAPE = (224,224,3)
BATCH_SIZE = 10
EPOCHS = 100
NUM_CLASSES = 2



def decode(filename):

 	feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([filename], num_epochs=EPOCHS)
    # Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
	image = tf.decode_raw(features['train/image'], tf.float32)
    
    # Cast label data into int32
	label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
	image = tf.reshape(image, [224, 224, 3])
	image, label = normalize(image, label)
    
	return image, label

def augment(image, label):
	"""Placeholder for data augmentation."""
	# OPTIONAL: Could reshape into a 28x28 image and apply distortions
	# here.  Since we are not applying any distortions in this
	# example, and the next step expects the image to be flattened
	# into a vector, we don't bother.
	return image, label

def normalize(image, label):
	"""Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	return image, label


def inputs(train, batch_size, num_epochs):
	"""Reads input data num_epochs times.
	Args:
	train: Selects between the training (True) and validation (False) data.
	batch_size: Number of examples per returned batch.
	num_epochs: Number of times to read the input data, or 0/None to
	   train forever.
	Returns:
	A tuple (images, labels), where:
	* images is a float tensor with shape [batch_size, (224,224,3)]
	  in the range [-0.5, 0.5].
	* labels is an int32 tensor with shape [batch_size] with the true label,
	  a number in the range [0, NUM_CLASSES).
	This function creates a one_shot_iterator, meaning that it will only iterate
	over the dataset once. On the other hand there is no special initialization
	required.
	"""
	

	if train:
		filename = 'train.tfrecords'
	else:
		filename = 'validation.tfrecords'

	image, label = decode(filename)
	images, labels = tf.train.batch([image, label], batch_size=BATCH_SIZE,  num_threads=1)
	# import ipdb; ipdb.set_trace()
	return images, labels


def run_training():


  
    image_batch_out, label_batch_out = inputs(
        train=True, batch_size=BATCH_SIZE, num_epochs=EPOCHS)
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int32, name="label_batch_offset")
    
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=NUM_CLASSES, on_value=1.0, off_value=0.0)
    logits_out = model_promo(image_batch_placeholder, 1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = label_batch_placeholder, logits = logits_out)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
    	 # Visualize the graph through tensorboard.
        # file_writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())
       
        # saver.restore(sess, "./output/model")
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        for i in range(100):
        	image_out, label_out, label_batch_one_hot_out = sess.run([image_batch, label_batch_out, label_batch_one_hot])
        	_, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

        	print("loss: ")
        	print(loss_out)
        	if(i%5 == 0):
        		saver.save(sess, "./output/model")

    	# coord.request_stop()
    	# coord.join(threads)
    	sess.close()

    
def main():
  run_training()


if __name__ == '__main__':
	main()