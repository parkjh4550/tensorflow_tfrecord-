########################
# input pipeline with  multi threads and TFRecord
#
# integration of all practice code

# reference : https://coolingoff.tistory.com/archive/201705
########################

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

NUM_EPOCHS = 10

########## 1. Prepare data
# Download mnist data
data_dir = "./mnist_data"       # directory where the mnist data will be saved.
data_sets = mnist.read_data_sets(data_dir,          #download the mnist data
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

# Split the data into "train", "test", "validation"  and save as TFRecord
data_splits = ["train", "test", "validation"]
save_dir = "./tfrecord_data"
for d in range(len(data_splits)):
    print("saving " + data_splits[d])
    data_set = data_sets[d]

    filename = os.path.join(save_dir, data_splits[d] + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=
                                                                      [data_set.images.shape[1]])),
            'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=
                                                                     [data_set.images.shape[2]])),
            'depth' : tf.train.Feature(int64_list=tf.train.Int64List(value=
                                                                     [data_set.images.shape[3]])),
            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=
                                                                     [int(data_set.labels[index])])),
            'image_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=
                                                                         [image]))
        }))

        writer.write(example.SerializeToString())
    writer.close()


############### 2. Load data with multi threads
# 2-1. Make Queue, Input Pipeline
filename = os.path.join(save_dir, "train.tfrecords")
filename_queue = tf.train.string_input_producer([filename], num_epochs=10) # generate filename queue

# Read filename using TFRecordReader()
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# parse the data feature
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# Get features and do type cast
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = features['label']
image.set_shape([784])

image = tf.cast(image, tf.float32) * (1./255) - 0.5
label = tf.cast(label, tf.int32)

# 2-2. Multi threads
# Shuffle and get batch
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000
)


############### 3. Model
# Simple DNN Model Define
W = tf.get_variable("W", [28*28, 10])
y_pred = tf.matmul(images_batch, W)     #calculate with batch and weights.

loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                       labels=labels_batch)

loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

################# 4. Train
# Make a Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)

# Make threads to input the data. Unlike the others, it nol only calls the symbol, but also "make threads".
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
print("#############generated threads")
print(threads)


# start training using threads
try:
    step=0
    while not coord.should_stop():
        step+=1
        sess.run([train_op])
        if step%500 ==0:
            loss_mean_val = sess.run([loss_mean])
            print("step : ", step)
            print("loss_mean : ", loss_mean_val)
except tf.errors.OutOfRangeError:
    # this occurs, if the queue is empty. It means all works are done.
    print('Done training for %d epochs, %d steps.' %(NUM_EPOCHS, step))
finally:
    # if finished, stop all threads
    coord.request_stop()

coord.join(threads) # wait until all threads stop
sess.close()