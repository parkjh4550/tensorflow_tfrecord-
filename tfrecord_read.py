import os
import tensorflow as tf
import numpy as np

save_dir = './tfrecord_data'
filename = os.path.join(save_dir, 'train.tfrecords')
print("\nload tfrecord from : ", filename)

record_iterator = tf.python_io.tf_record_iterator(filename)
seralized_img_example = next(record_iterator)

# parsing the data structure as Byte string
example = tf.train.Example()
example.ParseFromString(seralized_img_example)

print("\nassign each features from bytestring data")
image = example.features.feature['image_raw'].bytes_list.value[0]
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]

img_flat = np.fromstring(image, dtype=np.uint8)
img_reshaped = img_flat.reshape((height,width, -1))

print("\n========loaded img info")
print("flag img shape : ", img_flat.shape)
print("reshaped img shape : ", img_reshaped.shape)


