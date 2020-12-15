from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function
import numpy as np
from sys import argv
import tensorflow as tf
import glob
import os
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def cnn_model(features,labels,mode):
	l=6
	filter_size1 = 5
	s = 2
	input_layer = tf.reshape(features["x"],[-1,l,l,l,1])
	
	with tf.device('/gpu:0'):
		#layer 1
		conv1 = tf.layers.conv3d(inputs = input_layer,filters = filter_size1,kernel_size = [3,3,3],strides = s,padding = 'same',name=  "conv1",activation = tf.nn.relu,use_bias = True,kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))

#		conv1 = tf.layers.conv3d(inputs = input_layer,filters = filter_size1,kernel_size = [3,3,3],strides = s,padding = 'same',name=  "conv1",activation = tf.nn.relu,use_bias = True,kernel_initializer = tf.initializers.truncated_normal(mean = 0.0,stddev = 0.1,dtype = tf.float32,seed = None),bias_initializer = tf.constant_initializer(0.1))
	



		#dense layer
		conv2_flat = tf.reshape(conv1,[-1,int(l*l*l*filter_size1/(s*s*s))])
		dense = tf.layers.dense(inputs = conv2_flat,units = int(l*l*l*filter_size1/(s*s*s)),activation = tf.nn.relu,use_bias = True,kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))

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
			print(tf.trainable_variables())
			return tf.estimator.EstimatorSpec(mode = mode,loss=loss,train_op=train_op)

		#eval
		eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# predict_1.0559_720.dat

def main():
	binary,sample,size,ins_size_str= argv
	inst_size = int(ins_size_str)
	sample_numb = int(sample)
	l = int(size)
	filename = glob.glob("predict_*.dat")	
	res = {}
	res_for_amin = {}
	#load model
	run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(log_device_placement=True,device_count={'GPU': 0}))
	cnn_classifier = tf.estimator.Estimator(model_fn = cnn_model,model_dir="./train.parm")
	#log
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)
	
	fp_final = open("accu_%d.dat"%(l),"w")	
	for f in filename:
		temp = f.split('_')[1]
		print(temp)
		t = temp.split('.')[0]+'.'+temp.split('.')[1]
		name = "clean_%s"%(f)
		#read predicting data
		#checking
		predict_sample_size = sample_numb
		fpr = open(f,"r")
		fpw = open(name,"w")
		count = 0
		for s in range(0,predict_sample_size):
			cache = fpr.readline()
		#	if (cache == "" or cache == '\n' or cache.count('#')!=2):
			if (cache == ""):
				continue
		#	temp_cache = cache.split('#')
		#	if(temp_cache[0] == "" or temp_cache[1] == ""):
		#		continue
			fpw.write(cache)
			count = count+1
		predict_sample_size = count
	       	#checking
		fpr.close()
		fpw.close()
#		os.remove(f)
		
		predict_data = np.zeros((predict_sample_size,l,l,l,1),dtype = np.float32)
		predict_labels = np.zeros((predict_sample_size,1),np.int)
      		#read training ata
		fp = open(name,"r")	
		for s in range(0,predict_sample_size):
			print(s)
			buff = fp.readline()
		#	overlap = buff.split('#')[1].split(' ')
		#	lab = buff.split('#')[0]
		#	print(lab)
			overlap = buff.split('\n')[0].split(' ')
			for i in range(0,l):#
				for j in range(0,l):
					for k in range(0,l):
						predict_data[s,i,j,k,0] = overlap[i*l*l+j*l+k]

		os.remove(name)
		#build classifier
		run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(log_device_placement=True,device_count={'GPU': 0}))
		cnn_classifier = tf.estimator.Estimator(model_fn = cnn_model,model_dir="./train.parm")
	
		#log
		tensors_to_log = {"probabilities":"softmax_tensor"}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)
	
		#get values
	#	names = cnn_classifier.get_variable_names()
	#	print(names)
	#	print(cnn_classifier.get_variable_value("conv1/kernel"))	

		#evaluate
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":predict_data},y=predict_labels,num_epochs=1,shuffle= False, num_threads = 1)
		eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
		accu = eval_results["accuracy"]
		
		#record rsult
		if t in res:
			res[t].append(accu)
		else: 
			res[t] = [accu]
	
	#write file
	fp_final.write("T P\n")
	keys = list(res.keys())
	for i in range(0,inst_size):
		for t in keys:
			fp_final.write("%s %lf\n"%(t,res[t][i]))	
		

if __name__ == "__main__":
	main()


	
