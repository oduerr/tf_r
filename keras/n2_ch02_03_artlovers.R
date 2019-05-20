# https://github.com/oduerr/tf_r/blob/master/keras/Hello_Keras.R
## First Demonstration Notebook on using Keras

if (FALSE){ #First time only for installing stuff
  devtools::install_github("rstudio/keras") #you might need to install devtools
  library(keras)
  install_keras()
}

library(keras)

#######################
# Generation of the data
gen_image_with_bars <- function (size, num_bars, vertical = TRUE) {
  img = array(0, dim=c(size, size, 1))
  for (i in 1:num_bars){
    x = as.integer(runif(1,1,size))
    y = as.integer(runif(1,1,size))
    l = as.integer(runif(1,y,size))
    if (vertical) {
      img[y:l,x,1] = 255
    } else {
      img[x,y:l,1] = 255
    }
  }
  return(img/255)
}

size = 50
x = gen_image_with_bars(50,5)
image(1:size, 1:size, t(apply(x, 2, rev)), col=gray((0:255)/255))
x[1:size,1:size,1]

n_train =  1000
X_train = array(0, dim=c(n_train, size, size, 1))
for (i in 1:n_train/2) {
  X_train[i,1:size,1:size,1]=gen_image_with_bars(size,10)
}
for (i in n_train/2:n_train) {
  X_train[i,1:size,1:size,1]=gen_image_with_bars(size,10, vertical = FALSE)
}
y_train = c(rep(0,n_train/2),rep(1,n_train/2))

#########################
# Definition of the model 

model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(1, c(5,5), padding='same', input_shape = c(size,size,1)) %>% 
  layer_max_pooling_2d(c(size, size)) %>% 
  layer_flatten() %>% 
  layer_dense(2) %>% 
  layer_activation_softmax()

summary(model)

model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
 
history <- model %>% fit(
  X_train, y_train, 
  epochs = 50, batch_size = 128
) 

# Evaluation of the weights
d = model$get_weights()
d[[1]]
image(1:5, 1:5, t(apply(d[[1]], 2, rev)), col=gray((0:255)/255))
