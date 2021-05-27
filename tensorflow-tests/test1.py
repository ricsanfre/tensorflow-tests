import tensorflow as tf

x=tf.placeholder(tf.float32, name="X")
y=tf.placeholder(tf.float32,  name="Y")
u=tf.get_variable("U", shape=[], dtype=tf.float32,  initializer=tf.zeros_initializer)

z=u*x+y

#print (x)
#print (y)
#print (z)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer=tf.summary.FileWriter('.',  sess.graph)
#print("U",  u)

print(sess.run(z,  feed_dict={x:[1, 3],  y:[2, 4]}))
writer.close()
