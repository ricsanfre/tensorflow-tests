#import os.path
import tensorflow as tf
import math

LOGDIR = "/home/ricardo/CODE/tensorflow-tests/logs/"
DATADIR = "/home/ricardo/CODE/tensorflow-tests/data/"
### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=DATADIR, one_hot=True,  reshape=False, validation_size=0)

def fc_layer(input, size_in, size_out, name="fc",  activation="sigmoid",  pkeep=1.0):
  with tf.name_scope(name):
    # weights are initilialized ramdomly    
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    if activation == "sigmoid":
        act =  tf.nn.sigmoid(tf.matmul(input, w) + b)
    elif activation == "softmax":
        act = tf.nn.softmax(tf.matmul(input, w)+b)
    elif activation == "relu":
        act = tf.nn.relu(tf.matmul(input, w)+b)
    else:
        act= tf.matmul(input, w)+b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.dropout(act, pkeep)

def conv_layer(input, filter_size,  channel_in, channel_out, stride=1,  name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([filter_size, filter_size, channel_in, channel_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act 

    
def mnist_model_fc_5layers(activation_function,  hparam,  learning_rate_type, dropout):
    tf.reset_default_graph()
    sess = tf.Session()
    # Setup placeholders, input images, labels and learning rate
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

    tf.summary.image('input', x, 3)
    #Reshaping input images
    x_reshape = tf.reshape(x, [-1, 784])
    
    # Number neurons per layer
    layer_neurons1 =200
    layer_neurons2 = 100
    layer_neurons3 = 60
    layer_neurons4 = 30
    layer1 = fc_layer(x_reshape, 784, layer_neurons1, "layer1" , activation_function,  pkeep)
    layer2 = fc_layer(layer1, layer_neurons1, layer_neurons2,  "layer2",  activation_function,  pkeep )
    layer3 =  fc_layer(layer2,  layer_neurons2, layer_neurons3,  "layer3" , activation_function,  pkeep)
    layer4 =  fc_layer(layer3,  layer_neurons3, layer_neurons4,  "layer4",  activation_function, pkeep )
    logits =  fc_layer(layer4,  layer_neurons4, 10,  "logits",  "none" )
    
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(logits)
        tf.summary.histogram("preditions", prediction)
    
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(xent)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
   
    summ = tf.summary.merge_all()
    
    # Initializing variables
    sess.run(tf.global_variables_initializer())
    
   # TensorBoard - Initializing training and testing writers
    trainer_writer = tf.summary.FileWriter(LOGDIR+ "training_"+hparam)
    test_writer =  tf.summary.FileWriter(LOGDIR+ "testing_"+hparam)
    
    #Display Graph in Tensor Board
    trainer_writer.add_graph(sess.graph)
    for i in range(10001):
        batch = mnist.train.next_batch(100)
        if (learning_rate_type == "decay"):
            # learning rate decay
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        elif (learning_rate_type == "fixed"):
            learning_rate = 0.003
            
        if i % 20 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1],  pkeep:1.0})
            trainer_writer.add_summary(s, i)
        if i % 100 == 0:
            [test_accuracy,  s]= sess.run([accuracy, summ], feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024], pkeep:1.0})
            test_writer.add_summary(s,i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1],  lr: learning_rate,  pkeep:dropout})
    
def mnist_model_conv_5layers(activation_function,  hparam,  learning_rate_type, dropout):
    tf.reset_default_graph()
    sess = tf.Session()
    # Setup placeholders, input images, labels and learning rate
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

    tf.summary.image('input', x, 3)
    
    # Number neurons per layer
    layer1_channels = 6
    layer2_channels = 12
    layer3_channels = 24
    layer4_neurons = 200
    layer1 = conv_layer(x, 6,  1, layer1_channels , 1, "layer1_conv")
    layer2 = conv_layer(layer1, 5 ,  layer1_channels, layer2_channels, 2,  "layer2_conv")
    layer3 =  conv_layer(layer2, 4,  layer2_channels, layer3_channels,  2, "layer3_conv")
    #flatting layer3 output 
    flatten = tf.reshape (layer3, shape=[-1, 7*7*layer3_channels]  )
    layer4 =  fc_layer(flatten,  7*7*layer3_channels, layer4_neurons,  "layer4_fc",  activation_function, pkeep )
    logits =  fc_layer(layer4,  layer4_neurons, 10,  "logits",  "none" )
    
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(logits)
        tf.summary.histogram("preditions", prediction)
    
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(xent)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
   
    summ = tf.summary.merge_all()
    
    # Initializing variables
    sess.run(tf.global_variables_initializer())
    
   # TensorBoard - Initializing training and testing writers
    trainer_writer = tf.summary.FileWriter(LOGDIR+ "training_"+hparam)
    test_writer =  tf.summary.FileWriter(LOGDIR+ "testing_"+hparam)
    
    #Display Graph in Tensor Board
    trainer_writer.add_graph(sess.graph)
    for i in range(10001):
        batch = mnist.train.next_batch(100)
        if (learning_rate_type == "decay"):
            # learning rate decay
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        elif (learning_rate_type == "fixed"):
            learning_rate = 0.003
            
        if i % 20 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1],  pkeep:1.0})
            trainer_writer.add_summary(s, i)
        if i % 100 == 0:
            [test_accuracy,  s]= sess.run([accuracy, summ], feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024], pkeep:1.0})
            test_writer.add_summary(s,i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1],  lr: learning_rate,  pkeep:dropout})



def make_hparam_string(network_type,  activation_function,  learning_rate_type,  dropout):
  return "%s_%s_lr_%s_pkeep_%s" % (network_type, activation_function,  learning_rate_type,  dropout)

def main():
    print('Starting main')
#    network_types = ["fc", "conv"]
    network_types = ["conv"]
#    activation_functions = ["sigmoid",  "relu"]
    activation_functions = ["relu"]
#    learning_rate_types = ["fixed",  "decay"]
    learning_rate_types = [ "decay"]
    dropout_percentages = [1.0 , 0.75]
#    dropout_percentages = [1.0 ]
    for network_type in network_types:
        for activation_function in activation_functions:    
            for learning_rate_type in learning_rate_types:
                for dropout in dropout_percentages:
                    hparam = make_hparam_string(network_type,  activation_function,  learning_rate_type,  dropout)
                    print('Starting run for %s' % hparam)
                    if network_type == "fc":
                        # NN fullyconnected 5 layers
                        mnist_model_fc_5layers(activation_function,  hparam,  learning_rate_type,  dropout)
                    elif network_type == "conv":
                        mnist_model_conv_5layers(activation_function,  hparam,  learning_rate_type,  dropout)
        
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  main() 
