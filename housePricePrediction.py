"""
This is a basic TensorFlow Model that predicts prices of houses based on size
Created along with Pluralsight TensorFlow course.
"""
#pylint: disable=invalid-name

import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generating houses sized bwteen 1000 and 3500 sq ft
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from house size with random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# Plot generated house size and price
plt.plot(house_size, house_price, 'bx') # x-axis, y-axis bx = blue x
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()

# We need to normalize values to prevent under/overflows
def normalize(array):
    """normalize the data."""
    return (array - array.mean()) / array.std()

# split the data into training and testing groups
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_price)

# define testing data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_price_norm = normalize(test_house_price)
test_house_size_norm = normalize(test_house_size)

# set up tensorflow placeholders that get updated as we decend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# define the variables holding the size_factor and price we set during training
# We initialize them to random values based on normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name = "price_offset")

# 2. Define the operations for the predicting values - predicted price = (size_factor * house size) + price_offset
# we use the tensorflow operators instead of the normal signs, because it makes it super clear that we're doing
# things inside of the tensorflow workflow
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# 3. define the loss function (how much error) - Mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

# optimizer learning rate, the size of steps down the gradient
learning_rate = 0.1

#4. define a gradient descent optimizer that will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initializing the variables to avoid undefined errors
init = tf.global_variables_initializer()

# launch the graph in session
with tf.Session() as sess:
    sess.run(init)

    # how often to display training progress, and how many times to iterate
    display_every = 2
    num_training_iter = 50

    # keep iterating trainig data
    for iteration in range(num_training_iter):

        # fit all training data
        for(x, y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict={ tf_house_size: x, tf_price: y })

         # display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
        
    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')


   # Plot of training and test data, and learned regression
    
    # get values used to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()