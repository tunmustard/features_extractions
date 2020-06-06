import tensorflow as tf
import pandas as pd

mnist_train = pd.read_csv('MNIST_data/train.csv')
mnist_test = pd.read_csv('MNIST_data/test.csv')

y_train = mnist_train[['label']][:30000]
x_train = mnist_train[['pixel' + str(idx) for idx in range(784)]][:30000]

y_dev = mnist_train[['label']][30000:42000]
x_dev = mnist_train[['pixel' + str(idx) for idx in range(784)]][30000:42000]

x_test = mnist_test[['pixel' + str(idx) for idx in range(784)]]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_dev = scaler.transform(x_dev)
x_test = scaler.transform(x_test)


print(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape)

EPSILON = 1e-3

def DNN_BN(x, weights, beta, scale, activation_function = None):
    wx = tf.matmul(x, weights)
    mean, var = tf.nn.moments(wx, [0])
    bn = tf.nn.batch_normalization(wx, mean, var, beta, scale, EPSILON)
    if not activation_function:
        return bn
    else:
        return activation_function(bn)
    
def DNN(x, weights, biases, activation_function = None):
    wx = tf.matmul(x, weights)
    score = wx + biases
    if not activation_function:
        return score
    else:
        return activation_function(score)
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def scale_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
def get_center_loss(features, labels):
    with tf.variable_scope('center', reuse=True):
        centers = tf.get_variable('centers')
    
    len_features = features.get_shape()[1]
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_sum((features - centers_batch) ** 2, [1])
 
    return loss

def update_centers(features, labels, alpha):
    with tf.variable_scope('center', reuse=True):
        centers = tf.get_variable('centers')
    
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers = tf.scatter_sub(centers,labels, diff)
    
    return centers
    
    
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.int64, [None, 1])

ys_one_hot = tf.one_hot(ys, 10)
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('center'):
    centers = tf.get_variable('centers', [10, 1024], dtype=tf.float32,\
                          initializer=tf.constant_initializer(0), trainable=False)
#------CNN1-------#
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(xs, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#-------CNN2-------#
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#-------DNN------#
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
s_fc1 = scale_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = DNN(h_pool2_flat, W_fc1, b_fc1, tf.nn.relu)

center_loss = get_center_loss(h_fc1, ys)

update_centers = update_centers(h_fc1, ys, 0.5)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#-------DNN2-----#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#----------------#
softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=y_conv)

loss = tf.reduce_mean(softmax_loss + 0 * center_loss)

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

result = tf.argmax(y_conv,1)

ground_truth = tf.reshape(ys, [-1])

correct_prediction = tf.equal(result, ground_truth)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(60):
        print('LOSS, Softmax_loss, Center_loss', sess.run([loss, softmax_loss, center_loss], feed_dict = {xs: x_train, ys: y_train.values, keep_prob:1.0}))
        print('ACC@TRAIN:', sess.run(accuracy, feed_dict = {xs: x_train, ys: y_train.values, keep_prob:1.0}))
        print('ACC@DEV:', sess.run(accuracy, feed_dict = {xs: x_dev, ys: y_dev.values, keep_prob:1.0}))
        j = 0
        while j < 30000:       
            _, cen = sess.run([train_op, update_centers], feed_dict = {xs: x_train[j:j+1000], ys: y_train[j:j+1000].values, keep_prob:1.0})
            
            j += 1000  
    pd.DataFrame({"ImageId": range(1, len(x_test) + 1), "Label": sess.run(result, feed_dict = {xs: x_test, keep_prob:1.0})}).to_csv('MNIST_data/CNN.csv', index=False)