library(tensorflow)
slim = tf$contrib$slim #Poor mans import tensorflow.contrib.slim as slim

tf$reset_default_graph() # Better to start from scratch
# Resizing the images
images = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
imgs_scaled = tf$image$resize_images(images, shape(224,224))

# Definition of the network
library(magrittr) 
# The last layer is the fc8 Tensor holding the probability of the 1000 classes
fc8 = slim$conv2d(imgs_scaled, 64, shape(3,3), scope='vgg_16/conv1/conv1_1') %>% 
      slim$conv2d(64, shape(3,3), scope='vgg_16/conv1/conv1_2')  %>%
      slim$max_pool2d( shape(2, 2), scope='vgg_16/pool1')  %>%

      slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_1')  %>%
      slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_2')  %>%
      slim$max_pool2d( shape(2, 2), scope='vgg_16/pool2')  %>%

      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_1')  %>%
      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_2')  %>%
      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool3')  %>%

      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_1')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_2')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool4')  %>%

      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_1')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_2')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool5')  %>%

      slim$conv2d(4096, shape(7, 7), padding='VALID', scope='vgg_16/fc6')  %>%
      slim$conv2d(4096, shape(1, 1), scope='vgg_16/fc7') %>% 

      # Setting the activation_fn=NULL does not work, so we get a ReLU
      slim$conv2d(1000, shape(1, 1), scope='vgg_16/fc8')  %>%
      tf$squeeze(shape(1, 2), name='vgg_16/fc8/squeezed')


# Uncomment for debugging
# tf$train$SummaryWriter('/tmp/dumm/vgg16', tf$get_default_graph())$close()
# tf$get_default_graph()$get_operations()

variables_to_restore = slim$get_variables_to_restore()
for (dd in variables_to_restore){
  print(dd$name)
}

#d = slim$assign_from_checkpoint('/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt', variables_to_restore)

restorer = tf$train$Saver()
sess = tf$Session()
restorer$restore(sess, '/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt')

library(jpeg)
img1 <- readJPEG('apple.jpg')
d = dim(img1)
imgs = array(255*img1, dim = c(1, d[1], d[2], d[3]))

# Test comparison with python code
imgs[1,200,200:205,1] #In python Some pixels [201 202 198 188 189 185]
fc8_vals = sess$run(fc8, dict(images = imgs))
fc8_vals[1:5] #In python [-4.18968439  1.16550434 -1.50405121 -3.15936828 -3.20157099]
probs = exp(fc8_vals)/sum(exp(fc8_vals))
idx = sort.int(fc8_vals, index.return = TRUE, decreasing = TRUE)$ix[1:10]

library(readr)
names = read_delim("imagenet_classes.txt", "\t", escape_double = FALSE, trim_ws = TRUE,col_names = FALSE)
for (id in idx){
  cat(id, fc8_vals[id], names[id,][[1]],'\n')
}