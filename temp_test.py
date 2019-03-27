import tensorflow as tf

a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
lab = tf.losses.mean_squared_error(a,b)
l_a1 = tf.losses.mean_squared_error(a, tf.zeros(a.shape))
print(a)
print(b)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(lab))
    print(sess.run(l_a1))
