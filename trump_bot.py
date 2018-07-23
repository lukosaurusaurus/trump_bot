import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json, requests, random

data = open("input.txt",'r').read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
hidden_size = 128
learning_rate = 0.1
batch_size = 50
training_steps = 5000
display_step = 200
timesteps = 10
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

x = tf.placeholder("float",[None,timesteps,vocab_size])
y = tf.placeholder("float",[None,timesteps,vocab_size])
# inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]

weights = {
    'out':tf.Variable(tf.random_normal([hidden_size,vocab_size]))
}
biases = {
    'out':tf.Variable(tf.random_normal([vocab_size]))
} 
def RNN(x,weights,biases):
    print(x)
    # x = tf.reshape(x, [-1,timesteps,1])

    # x = tf.split(x,timesteps,vocab_size)
    x = tf.unstack(x,timesteps,1)
    
    rnn_cell = rnn.BasicLSTMCell(hidden_size)

    outputs,states = rnn.static_rnn(rnn_cell,x,dtype=tf.float32)

    print(len(outputs))
    return tf.matmul(outputs[-1], weights['out']) + biases['out']  

logits = RNN(x,weights,biases)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

# rnn_cell = rnn.MultiRNNCell([rnn.basicLSTMCell(hidden_size),rnn.basicLSTMCell(hidden_size)])

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        # [char_to_ix[ch] for ch in data[p:p+seq_length]]
        batch_x = tf.one_hot([char_to_ix[ch] for ch in data[step*batch_size*timesteps:(step+1)*batch_size*timesteps]],vocab_size).eval()
        batch_y = tf.one_hot([char_to_ix[ch] for ch in data[step*batch_size*timesteps+1:(step+1)*batch_size*timesteps+1]],vocab_size).eval()
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, vocab_size))
        batch_y = batch_y.reshape((batch_size, timesteps, vocab_size))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss, accuracy], feed_dict={x: batch_x,
                                                                 y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            test_len = 128
            xs = [tf.one_hot([random.randrange(0,vocab_size)],vocab_size)]
            test_sample = []
            for i in range(test_len):
                rnn_cell = rnn.BasicLSTMCell(hidden_size)
                outputs,states = rnn.static_rnn(rnn_cell,xs[i],dtype=tf.float32)
                logits = tf.matmul(outputs, weights['out']) + biases['out'] 
                prediction = tf.nn.softmax(logits)
                next_char = tf.one_hot(tf.argmax(prediction),vocab_size)
                xs.append(next_char)
                test_sample.append(ix_to_char(tf.argmax(next_char).eval()[0]))
                print(test_sample)

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))