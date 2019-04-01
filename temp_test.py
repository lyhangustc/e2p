import tensorflow as tf

a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
lab = tf.losses.mean_squared_error(a,b)
l_a1 = tf.losses.mean_squared_error(a, tf.zeros(a.shape))

with tf.variable_scope('V1') as scope:
	a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
with tf.variable_scope('V1', reuse=False) as scope:
	a3 = tf.get_variable('a1')
 
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(a1.name)
	print(sess.run(a1))
	print(a3.name)
	print(sess.run(a3))