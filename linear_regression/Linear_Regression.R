# This examples has been adapted taken from the RStudio website
library(ggplot2)
library(tensorflow)

# Creation of data 
N = 100
set.seed(42)
x_data <- runif(N, min=0, max=1)
y_data <- x_data * 0.1 + 0.3 + rnorm(N,0,0.01)
qplot(x_data, y_data) + geom_smooth(se=FALSE, method = 'lm')
lm(y_data ~ x_data) # <--- All we do in the next lines is replacing this

# We are fitting a*x+b to the data
tf$reset_default_graph() #<-- Quite important for R, to start from scratch writing in default graph
a <- tf$Variable(tf$ones(shape()), name='a')
b <- tf$Variable(tf$zeros(shape()), name='b')
x <- tf$placeholder('float32', shape(NULL), name='x_placeholder') #shape(N) would be also OK but we can leave it open
y <- tf$placeholder('float32', shape(NULL), name='y_placeholder') 
y_hat <- tf$scalar_mul(a, x) + b

# Minimize the mean squared errors.
loss <- tf$reduce_mean((y_hat - y) ^ 2, name='tot_loss')

# Looking at the graph
tf$train$SummaryWriter('/tmp/dumm/tf_graph_with_loss', tf$get_default_graph())$close()

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$initialize_all_variables())
res = sess$run(loss, feed_dict=dict(x = x_data, y = y_data))
print(res)

# Checking the result a = 1, b = 0
mean((1*x_data - y_data)^2)



optimizer <- tf$train$GradientDescentOptimizer(0.5) #<-- And here a miracle happens
train <- optimizer$minimize(loss) #Adds the loss op to the graph

tf$train$SummaryWriter('/tmp/dumm/tf_graph_with_loss_nun_wirklich', tf$get_default_graph())$close()
    
sess$run(tf$initialize_all_variables())
# Fit the line 
for (step in 1:201) {
  # idx = sample(1:N, 20) #Minibacth
  # sess$run(train, feed_dict=dict(x = x_data[idx], y = y_data[idx]))
  sess$run(train, feed_dict=dict(x = x_data, y = y_data))
  if (step %% 20 == 0) {
    cat(step, "-", 'b =', sess$run(b), 'a = ', sess$run(a),  "\n")
  }
}
sess$close()

