import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from sklearn.cross_validation  import train_test_split
from random import shuffle
import glob
import sys



#####
# load and shuffle data
#####

print('loading data and spilitting it to 60% train , 20% validation and 20% test')
shuffle_data = True
data_path = '/home/shiva/data/promotags/*.jpg'
address = glob.glob(data_path)
labels = [0 if 'non' in addr else 1 for addr in address] # 0 =  non_sales_tag , 1 = sales_tag

if shuffle_data:
  c = list(zip(address, labels))
  shuffle(c)
  address, labels = zip(*c)

train_addrs = address[0:int(0.6*len(address))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = address[int(0.6*len(address)):int(0.8*len(address))]
val_labels = labels[int(0.6*len(address)):int(0.8*len(address))]
test_addrs = address[int(0.8*len(address)):]
test_labels = labels[int(0.8*len(labels)):]


### Function to load images

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

### convert data to features

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




### writing data to tf record
train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    # print how many images are saved every 100 images
    if not i % 100:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()


# open the TFRecords file
val_filename = 'validation.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)
for i in range(len(val_addrs)):
    # print how many images are saved every 100 images
    if not i % 100:
        print 'Val data: {}/{}'.format(i, len(val_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(val_addrs[i])
    label = val_labels[i]
    # Create a feature
    feature = {'val/label': _int64_feature(label),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()


# open the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # print how many images are saved every 100 images
    if not i % 100:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(test_addrs[i])
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

