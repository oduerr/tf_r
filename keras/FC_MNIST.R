# https://github.com/oduerr/tf_r/blob/master/keras/Hello_Keras.R
## First Demonstration Notebook on using Keras

if (FALSE){ #First time only for installing stuff
  devtools::install_github("rstudio/keras") #you might need to install devtools
  library(keras)
  install_keras()
}

library(keras)
# Loading of the data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# rescale
x_train <- x_train / 255
#Q: Why did we do this?
x_test <- x_test / 255

# Visualization of the data
idx = 1
im = x_train[idx,,]
im
image(1:28, 1:28, t(apply(im, 2, rev)), col=gray((0:255)/255))
y_train[idx]

# One-hot encoding of the training data
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
dim(x_train)


model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "sigmoid", input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

y_hat = keras_predict(model, x_test) #gives probabilities
y_hat = predict_classes(model, x_test) #gives classes
mean(y_hat == mnist$test$y) #Accucary

########
# Tasks
# Do some experiments with the provided code. For all tasks below: compare the
# learning curves and evaluate the performance (accuracy) on the testset.
# 
# 
# 1) Add a dropout layer, with p=0.3, what is the accuracy you can reach? .
# 2) Add another layer and compare with your result
# 3) Change the activation to relu
# 4) Investigate some wrongly predicted images
# Optional:
# 5) Do the same analysis on fashion MNIST https://keras.rstudio.com/articles/tutorial_basic_classification.html








