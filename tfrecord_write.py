import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

data_dir = "./mnist_data"
save_dir = "./tfrecord_data"

# download dataset to "save_dir"
data_sets = mnist.read_data_sets(data_dir, dtype=tf.uint8, reshape=False, validation_size=1000)

# split data into "train, test, validation"
data_splits = ["train", "test", "validation"]
for d in range(len(data_splits)):
    print("\nsaving " + data_splits[d])
    data_set = data_sets[d]

    # set the tfwriter info
    filename = os.path.join(save_dir, data_splits[d]  + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    print("========Original image")
    print("type : ", type(data_set.images[0]))
    print("image : ", data_set.images[0].shape)

    print("========Byte string image")
    print("type : ", type(data_set.images[0].tostring()))

    for index in range(data_set.images.shape[0]):

        # data type change
        # image : numpy array ->  byte string
        image = data_set.images[index].tostring()

        #assign features in dictionary type
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=
                                       tf.train.Int64List(value=
                                                          [data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=
                                          tf.train.BytesList(value=
                                                             [image]))
        }))

        writer.write(example.SerializeToString())


    writer.close()