# Further information
# Google Blog Entry https://research.googleblog.com/2016/08/improving-inception-and-image.html
# Models to Train   https://github.com/tensorflow/models/blob/master/slim/README.md#Tuning
# Transfer Learning http://stackoverflow.com/questions/40350539/tfslim-problems-loading-saved-checkpoint-for-vgg16

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import numpy as np
from imagenet_classes import class_names



img1 = imread('apple.jpg')
print(img1.shape)
print("Some pixels {}".format(img1[199,199:205,0]))
#plt.imshow(img1)
#plt.show()

images = tf.placeholder(tf.float32, [None, None, None, 3])
imgs_scaled = tf.image.resize_images(images, (224,224))

fc8, _endpoints = slim.nets.vgg.vgg_16(imgs_scaled, is_training=False)
variables_to_restore = slim.get_variables_to_restore()
print('Number of variables to restore {}'.format(len(variables_to_restore)))
init_assign_op, init_feed_dict = slim.assign_from_checkpoint('/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt', variables_to_restore)

# tf.train.SummaryWriter('/tmp/dumm/vgg16_py', tf.get_default_graph()).close()

with tf.Session() as sess:
    sess.run(init_assign_op, init_feed_dict)
    prob = sess.run(fc8, {images:[img1]})[0]
    print("The first predictions {}".format(prob[0:5]))
    preds = (np.argsort(prob)[::-1])[0:10]
    for p in preds:
        print p, class_names[p], prob[p]

