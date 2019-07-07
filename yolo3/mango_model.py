"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose
from yolo3.boxes_utils import yolo_head, yolo_correct_boxes, yolo_boxes_and_scores, yolo_boxes_and_scores_ori_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
	"""Wrapper to set Darknet parameters for Convolution2D."""
	darknet_conv_kwargs = dict(kernel_regularizer=l2(5e-4))
	darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
	darknet_conv_kwargs.update(kwargs)
	return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky_input(*args, **kwargs):
	"""Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
	no_bias_kwargs = {'use_bias': True, 'input_shape': [416, 416, 3]}
	no_bias_kwargs.update(kwargs)
	return compose(
		DarknetConv2D(*args, **no_bias_kwargs),
		BatchNormalization(),
		LeakyReLU(alpha=0.1))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
	"""Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
	no_bias_kwargs = {'use_bias': True}
	no_bias_kwargs.update(kwargs)
	return compose(
		DarknetConv2D(*args, **no_bias_kwargs),
		BatchNormalization(),
		LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
	"""A series of resblocks starting with a downsampling Convolution2D"""
	# Darknet uses left and top padding instead of 'same' mode
	x = ZeroPadding2D(((1,0),(1,0)))(x)
	x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
	for i in range(num_blocks):
		y = compose(
				DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
				DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
		x = Add()([x,y])
	return x


def darknet_body(x):
	"""Darknent body having 52 Convolution2D layers"""
	x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
	x = resblock_body(x, 64, 1)
	x = resblock_body(x, 128, 2)
	x = resblock_body(x, 256, 8)
	x = resblock_body(x, 512, 8)
	x = resblock_body(x, 1024, 4)
	return x


def make_last_layers(x, num_filters, out_filters):
	"""6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
	x = compose(
			DarknetConv2D_BN_Leaky(num_filters, (1,1)),
			DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
			DarknetConv2D_BN_Leaky(num_filters, (1,1)),
			DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
			DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
	y = compose(
			DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
			DarknetConv2D(out_filters, (1,1)))(x)
	return x, y


def mango_yolo_body(inputs, num_anchors, num_classes):
	"""Create Tiny YOLO_v3 model CNN body in keras."""
	x1 = compose(
		DarknetConv2D_BN_Leaky(16, (3,3)),
		DarknetConv2D_BN_Leaky(16, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(32, (3,3)),
		DarknetConv2D_BN_Leaky(32, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(64, (3,3)),
		DarknetConv2D_BN_Leaky(64, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(128, (3,3)))(inputs)
	x2 = compose(
		DarknetConv2D_BN_Leaky(128, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(256, (3, 3)),
	)(x1)
	x3 = compose(
		DarknetConv2D_BN_Leaky(256, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(512, (3, 3))
	)(x2)
	y1 = compose(
		DarknetConv2D_BN_Leaky(1024, (3, 3)),
		DarknetConv2D_BN_Leaky(512, (3, 3)),
		DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))
	)(x3)
	x4 = compose(
		DarknetConv2D_BN_Leaky(256, (1,1)),
		UpSampling2D(2)
	)(x3)
	y2 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(256, (1,1)),
		DarknetConv2D_BN_Leaky(512, (3, 3)),
		DarknetConv2D_BN_Leaky(256, (1, 1)),
		DarknetConv2D(num_anchors*(num_classes+5), (1,1))
	)([x2, x4])
	x5 = compose(
		DarknetConv2D_BN_Leaky(128, (1, 1)),
		UpSampling2D(2)
	)(y2)
	y3 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(128, (1, 1)),
		DarknetConv2D_BN_Leaky(256, (3, 3)),
		DarknetConv2D_BN_Leaky(128, (1, 1)),
		DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))
	)([x5, x1])
	return Model(inputs, [y1, y2, y3])


def pruned_mango_yolo_body_cfg(inputs, num_anchors, num_classes, cfg):
	"""Create Tiny YOLO_v3 model CNN body in keras.
	"""
	cfg[13] = num_anchors * (num_classes + 5)
	cfg[18] = num_anchors * (num_classes + 5)
	cfg[31] = num_anchors * (num_classes + 5)
	x1 = compose(
		DarknetConv2D_BN_Leaky(cfg[0], (3,3)),
		DarknetConv2D_BN_Leaky(cfg[1], (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(cfg[2], (3,3)),
		DarknetConv2D_BN_Leaky(cfg[3], (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(cfg[4], (3,3)),
		DarknetConv2D_BN_Leaky(cfg[5], (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(cfg[6], (3,3)))(inputs)
	x2 = compose(
		DarknetConv2D_BN_Leaky(cfg[7], (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(cfg[8], (3, 3)),
	)(x1)
	x3 = compose(
		DarknetConv2D_BN_Leaky(cfg[9], (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(cfg[10], (3, 3))
	)(x2)
	y1 = compose(
		DarknetConv2D_BN_Leaky(cfg[11], (3, 3)),
		DarknetConv2D_BN_Leaky(cfg[12], (3, 3)),
		DarknetConv2D(cfg[13], (1, 1))
	)(x3)
	x4 = compose(
		DarknetConv2D_BN_Leaky(cfg[14], (1,1)),
		UpSampling2D(2)
	)(x3)
	y2 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(cfg[15], (1,1)),
		DarknetConv2D_BN_Leaky(cfg[16], (3, 3)),
		DarknetConv2D_BN_Leaky(cfg[17], (1, 1)),
		DarknetConv2D(cfg[18], (1,1))
	)([x2, x4])
	x5 = compose(
		DarknetConv2D_BN_Leaky(cfg[19], (1, 1)),
		UpSampling2D(2)
	)(y2)
	y3 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(cfg[20], (1, 1)),
		DarknetConv2D_BN_Leaky(cfg[21], (3, 3)),
		DarknetConv2D_BN_Leaky(cfg[22], (1, 1)),
		DarknetConv2D(cfg[23], (1, 1))
	)([x5, x1])
	return Model(inputs, [y1, y2, y3])


def pruned_mango_yolo_body(inputs, num_anchors, num_classes, cfg):
	"""Create Tiny YOLO_v3 model CNN body in keras.
	"""
	x1 = compose(
		DarknetConv2D_BN_Leaky(16, (3,3)),
		DarknetConv2D_BN_Leaky(16, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(32, (3,3)),
		DarknetConv2D_BN_Leaky(32, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(64, (3,3)),
		DarknetConv2D_BN_Leaky(64, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(64, (3,3)))(inputs)
	x2 = compose(
		DarknetConv2D_BN_Leaky(64, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(128, (3, 3)),
	)(x1)
	x3 = compose(
		DarknetConv2D_BN_Leaky(128, (3, 3), strides=2),
		DarknetConv2D_BN_Leaky(256, (3, 3))
	)(x2)
	y1 = compose(
		DarknetConv2D_BN_Leaky(256, (3, 3)),
		DarknetConv2D_BN_Leaky(64, (3, 3)),
		DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))
	)(x3)
	x4 = compose(
		DarknetConv2D_BN_Leaky(128, (1,1)),
		UpSampling2D(2)
	)(x3)
	y2 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(128, (1,1)),
		DarknetConv2D_BN_Leaky(256, (3, 3)),
		DarknetConv2D_BN_Leaky(64, (1, 1)),
		DarknetConv2D(num_anchors*(num_classes+5), (1,1))
	)([x2, x4])
	x5 = compose(
		DarknetConv2D_BN_Leaky(64, (1, 1)),
		UpSampling2D(2)
	)(y2)
	y3 = compose(
		Concatenate(),
		DarknetConv2D_BN_Leaky(64, (1, 1)),
		DarknetConv2D_BN_Leaky(128, (3, 3)),
		DarknetConv2D_BN_Leaky(64, (1, 1)),
		DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))
	)([x5, x1])
	return Model(inputs, [y1, y2, y3])


def yolo_eval(yolo_outputs,
			  anchors,
			  num_classes,
			  image_shape,
			  max_boxes=20,
			  score_threshold=.6,  # max_boxes=20, score_threshold=.6,iou_threshold=.5
			  iou_threshold=.5):
	"""Evaluate YOLO model on given input and return filtered boxes."""
	num_layers = len(yolo_outputs)
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers == 3 else [[3,4,5], [0,1,2]]  # default setting
	input_shape = K.shape(yolo_outputs[0])[1:3] * 32
	boxes = []
	box_scores = []
	for l in range(num_layers):
		_boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
			anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
		boxes.append(_boxes)
		box_scores.append(_box_scores)
	boxes = K.concatenate(boxes, axis=0)
	box_scores = K.concatenate(box_scores, axis=0)

	mask = box_scores >= score_threshold
	max_boxes_tensor = K.constant(max_boxes, dtype='int32')
	boxes_ = []
	scores_ = []
	classes_ = []
	for c in range(num_classes):
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
		nms_index = tf.image.non_max_suppression(
			class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
		class_boxes = K.gather(class_boxes, nms_index)
		class_box_scores = K.gather(class_box_scores, nms_index)
		classes = K.ones_like(class_box_scores, 'int32') * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
	boxes_ = K.concatenate(boxes_, axis=0)
	scores_ = K.concatenate(scores_, axis=0)
	classes_ = K.concatenate(classes_, axis=0)

	return boxes_, scores_, classes_


def yolo_eval_grad_am(yolo_outputs, yolo_model,
			  anchors,
			  num_classes,
			  image_shape,
			  max_boxes=20,
			  score_threshold=.6,  # max_boxes=20, score_threshold=.6,iou_threshold=.5
			  iou_threshold=.5):
	"""Evaluate YOLO model on given input and return filtered boxes."""
	num_layers = len(yolo_outputs)
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers == 3 else [[3,4,5], [0,1,2]]  # default setting
	input_shape = K.shape(yolo_outputs[0])[1:3] * 32
	boxes = []
	box_scores = []
	mask_L = []

	for l in range(num_layers):
		_boxes, _box_scores, output_scores_ori = yolo_boxes_and_scores_ori_shape(yolo_outputs[l],
			anchors[anchor_mask[l]], num_classes, input_shape, image_shape, score_threshold)
		boxes.append(_boxes)
		box_scores.append(_box_scores)
		mask_L.append(tf.cast(output_scores_ori >= score_threshold, tf.float32))

	boxes = K.concatenate(boxes, axis=0)
	box_scores = K.concatenate(box_scores, axis=0)

	mask = box_scores >= score_threshold

	yolo_outputs_masked_L = []
	for l in range(num_layers):
		mask_temp = mask_L[l]
		# yolo_output_temp = tf.reshape(yolo_outputs[l], mask_temp.get_shape())
		masked_yolo_output = tf.boolean_mask(yolo_outputs[l], mask_temp)
		yolo_outputs_masked_L.append(masked_yolo_output)

	# yolo_outputs_masked_L = []
	# for l in range(num_layers):
	#     mask_temp = mask_L[l]
	#     masked_yolo_output = tf.boolean_mask(yolo_outputs[l],
	#                                          (mask_temp[:, 32] + mask_temp[:, 47] + mask_temp[:, 49]) > 0)
	#     yolo_outputs_masked_L.append(masked_yolo_output)
	mask_temp1 = mask_L[1]
	want_mask = []
	# want_mask.append(tf.boolean_mask(yolo_outputs[1], (mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0))
	# want_mask.append((mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0)
	# want_mask.append(mask_temp1[:, 32] > 0)
	want_mask.append(mask_temp1)
	want_mask.append(yolo_outputs[1])

	conv_layer_outputs = []
	for layer in yolo_model.layers:
		if layer.__class__.__name__ == 'Conv2D':
			print(layer.name)
			conv_layer_outputs.append(layer.output)

	gradients = tf.gradients(yolo_outputs_masked_L, conv_layer_outputs)
	grad_ams = []
	for i_layer_num in range(len(conv_layer_outputs)):
		grad_am = conv_layer_outputs[i_layer_num] * gradients[i_layer_num]
		grad_am_reduce_sum = tf.reduce_max(grad_am, [0, 1, 2])
		grad_ams.append(grad_am_reduce_sum)
		# if grad_am_reduce_sum.shape[0].value > 40 and grad_am_reduce_sum.shape[0].value != 255:
		#     grad_ams.append(grad_am_reduce_sum)

	max_boxes_tensor = K.constant(max_boxes, dtype='int32')
	boxes_ = []
	scores_ = []
	classes_ = []
	for c in range(num_classes):
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
		nms_index = tf.image.non_max_suppression(
			class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
		class_boxes = K.gather(class_boxes, nms_index)
		class_box_scores = K.gather(class_box_scores, nms_index)
		classes = K.ones_like(class_box_scores, 'int32') * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
	boxes_ = K.concatenate(boxes_, axis=0)
	scores_ = K.concatenate(scores_, axis=0)
	classes_ = K.concatenate(classes_, axis=0)

	return boxes_, scores_, classes_, grad_ams, want_mask


def yolo_eval_grad_max(yolo_outputs, yolo_model,
			  anchors,
			  num_classes,
			  image_shape,
			  max_boxes=20,
			  score_threshold=.6,  # max_boxes=20, score_threshold=.6,iou_threshold=.5
			  iou_threshold=.5):
	"""Evaluate YOLO model on given input and return filtered boxes."""
	num_layers = len(yolo_outputs)
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers == 3 else [[3,4,5], [0,1,2]]  # default setting
	input_shape = K.shape(yolo_outputs[0])[1:3] * 32
	boxes = []
	box_scores = []
	mask_L = []
	for l in range(num_layers):
		_boxes, _box_scores, output_scores_ori = yolo_boxes_and_scores_ori_shape(yolo_outputs[l],
			anchors[anchor_mask[l]], num_classes, input_shape, image_shape, score_threshold)
		boxes.append(_boxes)
		box_scores.append(_box_scores)
		mask_L.append(tf.cast(output_scores_ori >= score_threshold, tf.float32))

	boxes = K.concatenate(boxes, axis=0)
	box_scores = K.concatenate(box_scores, axis=0)

	mask = box_scores >= score_threshold

	yolo_outputs_masked_L = []
	for l in range(num_layers):
		mask_temp = mask_L[l]
		# yolo_output_temp = tf.reshape(yolo_outputs[l], mask_temp.get_shape())
		masked_yolo_output = tf.boolean_mask(yolo_outputs[l], mask_temp)
		yolo_outputs_masked_L.append(masked_yolo_output)

	# yolo_outputs_masked_L = []
	# for l in range(num_layers):
	#     mask_temp = mask_L[l]
	#     masked_yolo_output = tf.boolean_mask(yolo_outputs[l],
	#                                          (mask_temp[:, 32] + mask_temp[:, 47] + mask_temp[:, 49]) > 0)
	#     yolo_outputs_masked_L.append(masked_yolo_output)
	mask_temp1 = mask_L[1]
	want_mask = []
	# want_mask.append(tf.boolean_mask(yolo_outputs[1], (mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0))
	# want_mask.append((mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0)
	# want_mask.append(mask_temp1[:, 32] > 0)
	want_mask.append(mask_temp1)
	want_mask.append(yolo_outputs[1])

	conv_layer_outputs = []
	for layer in yolo_model.layers:
		if layer.__class__.__name__ == 'Conv2D':
			print(layer.name)
			conv_layer_outputs.append(layer.output)

	gradients = tf.gradients(yolo_outputs_masked_L, conv_layer_outputs)
	grads = []
	for i_layer_num in range(len(conv_layer_outputs)):
		grad_reduce_max = tf.reduce_max(gradients[i_layer_num], [0, 1, 2])
		grads.append(grad_reduce_max)
		# if grad_am_reduce_sum.shape[0].value > 40 and grad_am_reduce_sum.shape[0].value != 255:
		#     grad_ams.append(grad_am_reduce_sum)

	max_boxes_tensor = K.constant(max_boxes, dtype='int32')
	boxes_ = []
	scores_ = []
	classes_ = []
	for c in range(num_classes):
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
		nms_index = tf.image.non_max_suppression(
			class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
		class_boxes = K.gather(class_boxes, nms_index)
		class_box_scores = K.gather(class_box_scores, nms_index)
		classes = K.ones_like(class_box_scores, 'int32') * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
	boxes_ = K.concatenate(boxes_, axis=0)
	scores_ = K.concatenate(scores_, axis=0)
	classes_ = K.concatenate(classes_, axis=0)

	return boxes_, scores_, classes_, grads, want_mask


def yolo_eval_grad_full_gradam(yolo_outputs, yolo_model,
			  anchors,
			  num_classes,
			  image_shape,
			  max_boxes=20,
			  score_threshold=.6,  # max_boxes=20, score_threshold=.6,iou_threshold=.5
			  iou_threshold=.5):
	"""Evaluate YOLO model on given input and return filtered boxes."""
	num_layers = len(yolo_outputs)
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]] # default setting
	input_shape = K.shape(yolo_outputs[0])[1:3] * 32
	boxes = []
	box_scores = []
	mask_L = []
	output_scores_ori_L = []
	for l in range(num_layers):
		_boxes, _box_scores, output_scores_ori = yolo_boxes_and_scores_ori_shape(yolo_outputs[l],
			anchors[anchor_mask[l]], num_classes, input_shape, image_shape, score_threshold)
		boxes.append(_boxes)
		box_scores.append(_box_scores)
		output_scores_ori_L.append(output_scores_ori)
		mask_L.append(tf.cast(output_scores_ori >= score_threshold, tf.float32))

	boxes = K.concatenate(boxes, axis=0)
	box_scores = K.concatenate(box_scores, axis=0)

	mask = box_scores >= score_threshold

	yolo_outputs_masked_L = []
	for l in range(num_layers):
		mask_temp = mask_L[l]
		masked_yolo_output = tf.boolean_mask(yolo_outputs[l], mask_temp)
		yolo_outputs_masked_L.append(masked_yolo_output)

	conv_layer_outputs = []
	for layer in yolo_model.layers:
		if layer.__class__.__name__ == 'Conv2D':
			print(layer.name)
			conv_layer_outputs.append(layer.output)

	gradients = tf.gradients(yolo_outputs_masked_L, conv_layer_outputs)
	grad_ams = []
	for i_layer_num in range(len(conv_layer_outputs)):
		grad_am = tf.nn.relu(conv_layer_outputs[i_layer_num] * gradients[i_layer_num])
		# grad_am_reduce_sum = tf.reduce_max(grad_am, [0, 1, 2])
		# if grad_am_reduce_sum.shape[0].value > 40 and grad_am_reduce_sum.shape[0].value != 255:
		grad_ams.append(grad_am)

	mask_temp1 = mask_L[1]
	want_mask = []
	# want_mask.append(tf.boolean_mask(yolo_outputs[1], (mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0))
	# want_mask.append((mask_temp1[:, 32] + mask_temp1[:, 47] + mask_temp1[:, 49]) > 0)
	# want_mask.append(mask_temp1[:, 32] > 0)
	want_mask.append(mask_temp1)
	want_mask.append(output_scores_ori_L)
	want_mask.append(box_scores)

	max_boxes_tensor = K.constant(max_boxes, dtype='int32')
	boxes_ = []
	scores_ = []
	classes_ = []
	for c in range(num_classes):
		# TODO: use keras backend instead of tf.
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
		nms_index = tf.image.non_max_suppression(
			class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
		class_boxes = K.gather(class_boxes, nms_index)
		class_box_scores = K.gather(class_box_scores, nms_index)
		classes = K.ones_like(class_box_scores, 'int32') * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
	boxes_ = K.concatenate(boxes_, axis=0)
	scores_ = K.concatenate(scores_, axis=0)
	classes_ = K.concatenate(classes_, axis=0)

	return boxes_, scores_, classes_, grad_ams, want_mask


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
	"""Preprocess true boxes to training input format

	Parameters
	----------
	true_boxes: array, shape=(m, T, 5)
		Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
	input_shape: array-like, hw, multiples of 32
	anchors: array, shape=(N, 2), wh
	num_classes: integer

	Returns
	-------
	y_true: list of array, shape like yolo_outputs, xywh are reletive value

	"""
	assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
	num_layers = len(anchors)//3 # default setting
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

	true_boxes = np.array(true_boxes, dtype='float32')
	input_shape = np.array(input_shape, dtype='int32')
	boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
	boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
	true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
	true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

	m = true_boxes.shape[0]
	grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
	y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
		dtype='float32') for l in range(num_layers)]

	# Expand dim to apply broadcasting.
	anchors = np.expand_dims(anchors, 0)
	anchor_maxes = anchors / 2.
	anchor_mins = -anchor_maxes
	valid_mask = boxes_wh[..., 0]>0

	for b in range(m):
		# Discard zero rows.
		wh = boxes_wh[b, valid_mask[b]]
		if len(wh)==0: continue
		# Expand dim to apply broadcasting.
		wh = np.expand_dims(wh, -2)
		box_maxes = wh / 2.
		box_mins = -box_maxes

		intersect_mins = np.maximum(box_mins, anchor_mins)
		intersect_maxes = np.minimum(box_maxes, anchor_maxes)
		intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
		intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
		box_area = wh[..., 0] * wh[..., 1]
		anchor_area = anchors[..., 0] * anchors[..., 1]
		iou = intersect_area / (box_area + anchor_area - intersect_area)

		# Find best anchor for each true box
		best_anchor = np.argmax(iou, axis=-1)

		for t, n in enumerate(best_anchor):
			for l in range(num_layers):
				if n in anchor_mask[l]:
					i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
					j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
					k = anchor_mask[l].index(n)
					c = true_boxes[b,t, 4].astype('int32')
					y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
					y_true[l][b, j, i, k, 4] = 1
					y_true[l][b, j, i, k, 5+c] = 1

	return y_true


def box_iou(b1, b2):
	"""Return iou tensor

	Parameters
	----------
	b1: tensor, shape=(i1,...,iN, 4), xywh
	b2: tensor, shape=(j, 4), xywh

	Returns
	-------
	iou: tensor, shape=(i1,...,iN, j)

	"""

	# Expand dim to apply broadcasting.
	b1 = K.expand_dims(b1, -2)
	b1_xy = b1[..., :2]
	b1_wh = b1[..., 2:4]
	b1_wh_half = b1_wh/2.
	b1_mins = b1_xy - b1_wh_half
	b1_maxes = b1_xy + b1_wh_half

	# Expand dim to apply broadcasting.
	b2 = K.expand_dims(b2, 0)
	b2_xy = b2[..., :2]
	b2_wh = b2[..., 2:4]
	b2_wh_half = b2_wh/2.
	b2_mins = b2_xy - b2_wh_half
	b2_maxes = b2_xy + b2_wh_half

	intersect_mins = K.maximum(b1_mins, b2_mins)
	intersect_maxes = K.minimum(b1_maxes, b2_maxes)
	intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
	b1_area = b1_wh[..., 0] * b1_wh[..., 1]
	b2_area = b2_wh[..., 0] * b2_wh[..., 1]
	iou = intersect_area / (b1_area + b2_area - intersect_area)

	return iou


def compute_iou(rec1, rec2):
	"""
	computing IoU
	:param rec1: (y0, x0, y1, x1), which reflects
			(top_true, left_true, bottom_true, right_true)
	:param rec2: (y0, x0, y1, x1)
	:return: scala value of IoU
	"""
	# computing area of each rectangles
	S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
	S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

	# computing the sum_area
	sum_area = S_rec1 + S_rec2

	# find the each edge of intersect rectangle
	left_line = max(rec1[1], rec2[1])
	right_line = min(rec1[3], rec2[3])
	top_line = max(rec1[0], rec2[0])
	bottom_line = min(rec1[2], rec2[2])

	# judge if there is an intersect
	if left_line >= right_line or top_line >= bottom_line:
		return 0
	else:
		intersect = (right_line - left_line) * (bottom_line - top_line)
		return intersect / (sum_area - intersect)


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
	"""Return yolo_loss tensor

	Parameters
	----------
	:param args:
	:param ignore_thresh: 0.5
	:param num_classes: integer
	:param anchors: array, shape=(N, 2), wh
	:param print_loss: False

	Returns
	-------
	loss: tensor, shape=(1,)

	"""
	num_layers = len(anchors)//3 # default setting

	# yolo_outputs is model's output [y1, y2]
	yolo_outputs = args[:num_layers]
	# y_true is same shape with yolo_outputs
	# y_true is two tensors' list [tensor1: y1(26, 26, 3, 6), tensor1: y2(13, 13, 3, 6)]
	y_true = args[num_layers:]

	# I changed mask index here
	anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
	# anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
	input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
	grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
	loss = 0
	m = K.shape(yolo_outputs[0])[0] # batch size, tensor
	mf = K.cast(m, K.dtype(yolo_outputs[0]))

	for l in range(num_layers):
		object_mask = y_true[l][..., 4:5]
		true_class_probs = y_true[l][..., 5:]

		grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
			 anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
		pred_box = K.concatenate([pred_xy, pred_wh])

		# Darknet raw box to calculate loss.
		raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
		raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
		raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf

		# 2 - true_w * true_h
		box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

		# Find ignore mask, iterate over each of batch.
		ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
		object_mask_bool = K.cast(object_mask, 'bool')

		def loop_body(b, ignore_mask):
			true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
			iou = box_iou(pred_box[b], true_box)
			best_iou = K.max(iou, axis=-1)
			ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
			return b+1, ignore_mask

		_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
		ignore_mask = ignore_mask.stack()
		ignore_mask = K.expand_dims(ignore_mask, -1)

		# K.binary_crossentropy is helpful to avoid exp overflow.
		xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
		wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
		confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
			(1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
		class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

		xy_loss = K.sum(xy_loss) / mf
		wh_loss = K.sum(wh_loss) / mf
		confidence_loss = K.sum(confidence_loss) / mf
		class_loss = K.sum(class_loss) / mf
		loss += xy_loss + wh_loss + confidence_loss + class_loss
		if print_loss:
			loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
	return loss
