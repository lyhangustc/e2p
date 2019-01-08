import tensorflow as tf
import numpy as np

"""
sess = tf.Session()
batch_size =16 
class_num = 1000
logits_scale = 10.0

labels = tf.cast(tf.random_uniform([batch_size]) * class_num, dtype=tf.int32)
onehot_labels =  tf.one_hot(labels, class_num)
logits = (tf.random_uniform([batch_size, class_num]) - 1.0) * 10.0

fake_one_hot = (tf.one_hot(labels, class_num) - 0.5) * 4.0
def softmax_ce(logits, onehot_labels):
    prob = tf.nn.softmax(logits=logits)
    loss = -tf.reduce_sum(onehot_labels * tf.log(prob)) /batch_size
    return loss


loss_tf = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=onehot_labels)
loss_fake_tf = tf.losses.softmax_cross_entropy(logits=fake_one_hot, onehot_labels=onehot_labels)

loss = softmax_ce(logits=logits, onehot_labels=onehot_labels)
loss_fake = softmax_ce(logits=fake_one_hot, onehot_labels=onehot_labels)

print(sess.run([loss_tf,loss_fake_tf, loss, loss_fake]))
print("END")
"""

logits = np.loadtxt('logits')
onehot_labels = np.loadtxt('onehot_labels')

logits_t = tf.placeholder(tf.float32, shape=(8, 10))
onehot_labels_t = tf.placeholder(tf.float32, shape=(8, 10))
loss = tf.losses.softmax_cross_entropy(logits=logits_t, onehot_labels=onehot_labels_t)
sess = tf.Session()
l = sess.run(loss, feed_dict={logits_t:logits, onehot_labels_t:onehot_labels})

print(l)
print(logits.shape, onehot_labels.shape)
print(logits.max(), logits.min())
print(onehot_labels.max(), onehot_labels.min())
