## First Demonstration Notebook on using Keras

if (FALSE){ #First time only for installing stuff
  devtools::install_github("rstudio/keras") #you might need to install devtools
  library(keras)
  install_keras()
}

library(keras)
boston_housing <- dataset_boston_housing()
X_train = scale(boston_housing$train$x)
y_train = boston_housing$train$y

X_test = scale(boston_housing$test$x) #Strictly one should ue mean and std from training
y_test = boston_housing$test$y

if (FALSE){
  dumm1 = keras_model_sequential() 
  model = layer_dense(dumm1, units = 1, input_shape = dim(X_train)[2])
}
# In DL this would be repeated several times. Nice syntactic sugar, the pipes
model = keras_model_sequential() %>% 
  layer_dense(units = 1, input_shape = dim(X_train)[2]) #We need to define the imput shape
  #First argument is dumm1
summary(model)
# Q: why do we have 14 trainable parameters

model = compile(model,loss='mse', optimizer = optimizer_rmsprop(0.2))
hist = fit(model, X_train, y_train, epochs = 100, validation_split = 0.02, verbose=0, shuffle = TRUE)
#Q: What is epochs what is validation_split
library(ggplot2)
plot(hist) + ylim(0,500)

y_hat_test = predict(model, X_test)
sqrt(mean((y_hat_test[,1] - y_test)^2))
#model$get_weights()

## Traditional R
train = data.frame(X_train)
train$y=y_train
l = lm(y ~ ., train) #Similar R
xt = data.frame(X_test)
y_hat_lm = predict(l, newdata = xt)
sqrt(mean((y_hat_lm - y_test)^2))

# A first deep learning model. Stack more Layers!
model = keras_model_sequential() %>% 
  layer_dense(units = 64, input_shape = dim(X_train)[2], activation = 'relu') %>%  #We need to define the imput shape
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 1)  

summary(model) 
model = compile(model,loss='mse', optimizer = optimizer_rmsprop())
hist = fit(model, X_train, y_train, epochs = 500, validation_split = 0.02, verbose=0, shuffle = TRUE)
#Q: What is epochs what is validation_split
plot(hist) + ylim(0,20)
y_hat_test = predict(model, X_test)
sqrt(mean((y_hat_test[,1] - y_test)^2))
