import tensorflow as tf
import numpy as np
import ops

x1 = tf.constant([[0.,1.,1.,1.], [1.,1.,1.,1.], [0.,1.,1.,1.]])
x2 = tf.constant([[0,2.,1,1], [1,1,1,1], [0,1,1,1]])
x = tf.multiply(x1,x2)

x = tf.greater(x, tf.ones(tf.shape(x)) * 1e-10)
y = ops.distance_transform(x)
print(y.get_shape())
y = tf.reshape(y, [3, 4])
print(y.get_shape())
sess = tf.Session()
print(sess.run(x))
print(sess.run(y))
        