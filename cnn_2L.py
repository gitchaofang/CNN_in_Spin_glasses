from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function
import numpy as np
from sys import argv
import tensorflow as tf
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def cnn_model(features,labels,mode):
	l=6
	filter_size1 = 7
	s1  = 2
	filter_size2 = 7
	s2 = 1
	input_layer = tf.reshape(features["x"],[-1,l,l,l,1])
	
	with tf.device('/gpu:0'):
		#layer 1
		conv1 = tf.layers.conv3d(inputs = input_layer,filters = filter_size1,kernel_size = [3,3,3],strides = s1,padding = 'same',activation = tf.nn.relu)
		#pool1 = tf.layers.average_pooling3d (inputs = conv1,pool_size = (2,2,2),strides = (2,2,2),padding = 'valid')
		pool1 = tf.layers.max_pooling3d(inputs = conv1,pool_size = (2,2,2),strides = 1,padding = 'same')
	
		#layer 2
		conv2 = tf.layers.conv3d(inputs = pool1,filters = filter_size2,kernel_size = [2,2,2],strides = s2,padding = 'same',activation = tf.nn.relu)

		#dense layer
		conv2_flat = tf.reshape(conv2,[-1,int(l*l*l*filter_size2/(s1*s1*s1*s2*s2*s2))])
		dense = tf.layers.dense(inputs = conv2_flat,units = l*l*l,activation = tf.nn.relu)

		#dropout
		dropout = tf.layers.dropout(inputs = dense,rate = 0.2,training = mode == tf.estimator.ModeKeys.TRAIN)
		#logits layer
		logits = tf.layers.dense(inputs = dropout,units = 2)
	
		predictions = {"classes":tf.argmax(input = logits,axis=1),"prob":tf.nn.softmax(logits,name = "softmax_tensor")}
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode = mode,predictions = predictions)

		#loss
		loss = tf.losses.sparse_softmax_cross_entropy(labels = labels,logits = logits)

		#train
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
			train_op = optimizer.minimize(loss = loss,global_step = tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode = mode,loss=loss,train_op=train_op)

		#eval
		eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main():
	binary,sample,size= argv
	sample_numb = int(sample)
	l = int(size)
	fp = open("train.dat","r")
	train_data = np.zeros((sample_numb,l*l*l),dtype = np.float32)
	train_labels = np.zeros((sample_numb,1),np.int)
	#read data
	for s in range(0,sample_numb):
	#	print(s)
		buff = fp.readline()
		overlap = buff.split('#')[0].split(' ')
		lab = buff.split('#')[1]
		for i in range(0,l*l*l):
			train_data[s,i] = float(overlap[i])
			train_labels[s,0] = float(lab)
	
	#check
#	for s in range (0,s):
#		for i in range(0,l*l*l):
#			print("%d "%(train_data[s,i]),end = '')
#		print(train_labels[s,0])

	

	#build classifier
	run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(log_device_placement=True,
                                      device_count={'GPU': 0}))

	cnn_classifier = tf.estimator.Estimator(model_fn = cnn_model)
	
	#log
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

	
	#train
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x":train_data},y=train_labels,batch_size=100,num_epochs = 1,shuffle = True,num_threads = 1)
	cnn_classifier.train(input_fn = train_input_fn,steps = 10000,hooks = [logging_hook])

	#evaluate
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":train_data},y=train_labels,num_epochs=1,shuffle= False, num_threads = 1)
	eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)






if __name__ == "__main__":
	main()


	
