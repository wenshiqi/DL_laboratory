# coding:utf-8
import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
trainning_iters = 100000
batch_size = 128

n_inputs = 28 # 图片大小28*28
n_steps = 28 # 时间步
n_hidden_units = 128
n_classes = 10 # 分类类别数目


#todo# 为什么要这样设置shape
x = tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,shape=[None,n_classes])



## 设置权重


#todo# get_variable和variable的区别是啥
weights = {
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X,weights,biases,tag='lstm'):
    # X(128 batch, 28 steps, 28 inputs) => (128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    # ==>(128 batch * 28 steps, 28 hidden)
    X_in = tf.matmul(X, weights['in'])+biases['in']

    #todo# 这里x_in的维度是128，28，128么，对于维度变化真的很容易懵
    X_in = tf.reshape(X_in,[-1,n_steps, n_hidden_units])

    if tag == 'lstm':
        #cell
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias =1.0, state_is_tuple = True)
        _init_state = cell.zero_state(batch_size, dtype=tf.float32)

        # outputs 存放每一个时间步的h,states存放最后一个时间步的c和h
        outputs, states = tf.nn.dynamic_rnn(cell, X_in, initial_state=_init_state, time_major=False)
        results = tf.matmul(states[1], weights['out']) + biases['out']
        return results

    elif tag == 'gru':
        cell = tf.contrib.rnn.GRUCell(n_hidden_units)
        _init_state = cell.zero_state(batch_size, dtype=tf.float32)

        # outputs 存放每一个时间步的h,states存放最后一个时间步的h
        outputs, states = tf.nn.dynamic_rnn(cell, X_in, initial_state=_init_state, time_major=False)
        results = tf.matmul(states, weights['out']) + biases['out']
        return results


pred = RNN(x, weights, biases, tag='gru')
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size<trainning_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step % 20 ==0:
            print (sess.run(accuracy, feed_dict={
                x:batch_xs,
                y:batch_ys
            }))
        step += 1




