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

restorer = tf$train$Saver()
sess = tf$Session()
restorer$restore(sess, '/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt')

library(jpeg)

img1 <- readJPEG('poodle.jpg')
d = dim(img1)
imgs = array(255*img1, dim = c(1, d[1], d[2], d[3]))

# Test comparison with python code
imgs[1,200,200:205,1] #In python Some pixels [152 153 151 150 152 155]
fc8_vals = sess$run(fc8, dict(images = imgs))
fc8_vals[1:5] #In python [-2.86833096  0.7060132  -1.32027602 -0.61107934 -1.67312801]
probs = exp(fc8_vals)/sum(exp(fc8_vals))

idx = sort.int(fc8_vals, index.return = TRUE, decreasing = TRUE)$ix[1:5]

# Reading the class names
library(readr)
names = read_delim("imagenet_classes.txt", "\t", escape_double = FALSE, trim_ws = TRUE,col_names = FALSE)

### Graph
library(grid)
g = rasterGrob(img1, interpolate=TRUE) 
text = ""
for (id in idx) {
  text = paste0(text, names[id,][[1]], " ", round(probs[id],5), "\n") 
}
ggplot(data.frame(d=1:3)) + annotation_custom(g) + 
  annotate('text',x=0.05,y=0.05,label=text, size=7, hjust = 0, vjust=0, color='gray90') + xlim(0,1) + ylim(0,1) 
