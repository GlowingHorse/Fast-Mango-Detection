"""
rewrite function “create_pruned_tiny_model_with_cfg” to achieve FLOPs computing
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from yolo3.model import yolo_body, yolo_loss, tiny_yolo_body_with_cfg


def _main():
	log_dir = 'logs/006prune_net4.7_for_train/001/'
	classes_path = 'model_data/mango_classes.txt'
	anchors_path = 'model_data/tiny_yolo_mango_anchors.txt'

	sort_idx_L = np.load(log_dir + "grad_am_sort_idx_L.npy", allow_pickle=True)
	class_names = get_classes(classes_path)
	num_classes = len(class_names)
	anchors = get_anchors(anchors_path)

	input_shape = (512, 512)
	conv_lyr_output_dim = [16,
						   32,
						   64,
						   128,
						   256,
						   512,
						   1024,
						   256,
						   128,
						   512,
						   256]

	# #  6.3Mb
	# prune_percentages = [0,
	# 					0,
	# 					0,
	# 					0.4,
	# 					0.4,
	# 					0.5,
	# 					0.7,
	# 					0.6,
	# 					0.8,
	# 					0.8,
	# 					0.6]

	# 4.6Mb
	prune_percentages = [0,
						 0,
						 0,
						 0.5,
						 0.5,
						 0.5,
						 0.75,
						 0.75,
						 0.75,
						 0.875,
						 0.875]
	cfg = []
	for i_cfg in range(11):
		sort_idx = sort_idx_L[i_cfg]
		prune_percentage = prune_percentages[i_cfg]
		prune_num = int(len(sort_idx)) - int(len(sort_idx) * prune_percentage)
		cfg.append(prune_num)

	# run_meta = tf.RunMetadata()
	with tf.Session(graph=tf.Graph()) as sess:
		K.set_session(sess)
		is_tiny_version = len(anchors) == 6  # default setting
		if is_tiny_version:
			model = create_pruned_tiny_model_with_cfg(input_shape, anchors, num_classes, cfg=cfg)
		else:
			model = create_model(input_shape, anchors, num_classes)

		opts = tf.profiler.ProfileOptionBuilder.float_operation()
		flops = tf.profiler.profile(sess.graph, cmd='op', options=opts)

		opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
		params = tf.profiler.profile(sess.graph, cmd='op', options=opts)
		# model.save_weights(log_dir + 'No_train_weights_MangoYolo.h5')
		print("{:,} ---".format(flops.total_float_ops))
		print("\n")
		print("{:,} ---".format(params.total_parameters))
		print("\n")
		# plot_model(model, to_file='pruned_yolov3tiny_model.png', show_shapes=True)


	# Further training if needed.


def get_classes(classes_path):
	'''loads the classes'''
	with open(classes_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names


def get_anchors(anchors_path):
	'''loads the anchors from a file'''
	with open(anchors_path) as f:
		anchors = f.readline()
	anchors = [float(x) for x in anchors.split(',')]
	return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes):
	'''create the training model'''
	# K.clear_session()  # get a new session
	h, w = input_shape

	image_tensor = K.ones(shape=(1, h, w, 3))
	image_input = Input(shape=(h, w, 3), name="shirui_input_0", tensor=image_tensor)
	# image_input = Input(shape=(h, w, 3))
	num_anchors = len(anchors)

	y_true = []
	for l in range(3):
		y_true_tensor = K.ones(shape=(1, h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],
							   num_anchors//3, num_classes+5))
		y_true_input = Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],
							num_anchors//3, num_classes+5), tensor=y_true_tensor)
		y_true.append(y_true_input)
	model_body = yolo_body(image_input, num_anchors // 3, num_classes)
	# model_body = mango_yolo_body(image_input, num_anchors//3, num_classes)
	print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

	model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
		arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
		[*model_body.output, *y_true])
	model = Model([model_body.input, *y_true], model_loss)

	return model


def create_pruned_tiny_model_with_cfg(input_shape, anchors, num_classes, cfg=None):
	'''create the training model, for Tiny YOLOv3'''
	# K.clear_session() # get a new session
	h, w = input_shape

	image_tensor = K.ones(shape=(1, h, w, 3))
	image_input = Input(shape=(h, w, 3), name="user_input_0", tensor=image_tensor)
	# image_input = Input(shape=(h, w, 3))
	num_anchors = len(anchors)

	y_true = []
	for l in range(2):
		y_true_tensor = K.ones(shape=(1, h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
							   num_anchors//2, num_classes+5))
		y_true_input = Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
							num_anchors//2, num_classes+5), tensor=y_true_tensor)
		y_true.append(y_true_input)
	model_body = tiny_yolo_body_with_cfg(image_input, num_anchors//2, num_classes, cfg)
	print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

	model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
		arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
		[*model_body.output, *y_true])
	model = Model([model_body.input, *y_true], model_loss)

	return model


if __name__ == '__main__':
	_main()
