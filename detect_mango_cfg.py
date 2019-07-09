"""
计算coco数据集中苹果和橘子类别的f1 score

"""
import sys
import argparse
from yolo3.yolo_cfg import YOLO
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os
from yolo3.model import compute_iou


def detect_img(yolo):
	train_annotation_path = './data_annotation/2007_test.txt'
	with open(train_annotation_path) as f:
			test_lines = f.readlines()

	TPFN = 0.0
	TPFP = 0.0
	TP = 0.0

	# hsv_tuples = [(1, 1., 1.)]
	# colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	# colors = list(
	# 		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
	# 			colors))

	for i_img_idx in range(len(test_lines)):
		img_annotation = test_lines[i_img_idx].split(" ")
		img_name = img_annotation[0]
		boxes = img_annotation[1:]
		true_box = []
		for box in boxes:
			true_box.append(np.array(list(map(int, box.split(',')))))

		# image_show = Image.open(img_name)
		# draw = ImageDraw.Draw(image_show)
		#
		# for i_true_box in range(len(true_box)):
		# 		left_true, top_true, right_true, bottom_true, class_label = true_box[i_true_box]
		# 		for i in range(3):
		# 				draw.rectangle(
		# 						[left_true + i, top_true + i, right_true - i, bottom_true - i],
		# 						outline=colors[0])
		# del draw
		# pix101 = np.array(image_show)
		# fig101 = plt.figure(101)
		# plt.imshow(pix101)
		# plt.show(block=False)

		image = Image.open(img_name)
		r_image, out_boxes = yolo.detect_image(image)
		pix = np.array(r_image)
		# _, save_img_name = os.path.split(img_name)
		# r_image.save('./gen_imgs/'+str(i_img_idx)+save_img_name)

		fig102 = plt.figure(102)
		plt.imshow(pix)
		plt.show(block=False)

		TP_temp = 0
		for i_true_box in range(len(true_box)):
			left_true, top_true, right_true, bottom_true, class_label = true_box[i_true_box]
			true_box_iou = (top_true, left_true, bottom_true, right_true)
			for i_out_boxes in range(len(out_boxes)):
				out_box_iou = tuple(out_boxes[i_out_boxes])
				iou_score = compute_iou(true_box_iou, out_box_iou)
				if iou_score > 0.5:
					TP = TP+1
					TP_temp = TP_temp + 1
					# remove used out_box
					out_boxes[i_out_boxes] = 0, 0, 0, 0
					break

		TPFP = TPFP + len(out_boxes)
		TPFN = TPFN + len(true_box)
		print("True box num: {}, detected box num: {}, TruePos box num: {}".format(len(true_box), len(out_boxes), TP_temp))
		print('TP : {}, TPFP :{}, TPFN :{} '.format(TP, TPFP, TPFN))
		print('\n')
	recall = TP / TPFN
	precision = TP / TPFP
	f1score = 2 * recall * precision / (recall + precision)
	print("F1score : {}".format(f1score))
	# plt.close(fig101)
	plt.close(fig102)
	yolo.close_session()


def _main():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	conv_lyr_idx = ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6",
			"conv2d_7", "conv2d_8", "conv2d_11", "conv2d_9", "conv2d_12"]

	conv_lyr_output_dim = [16, 32, 64, 128, 256, 512,
			       1024, 256, 128, 512, 256]
	# #  003 5.2Mb
	# prune_percentages = [0.2, 0, 0, 0.5, 0.5, 0.5,
	# 		0.75, 0.6, 0.75, 0.75, 0.75]

	# #  5.3Mb
	prune_percentages = [0, 0, 0, 0.5, 0.5, 0.5,
			     0.75, 0.5, 0.75, 0.75, 0.75]

	# #  6.3Mb
	# prune_percentages =   [0,    0,          0,          0.4,        0.4,        0.5,
	#                        0.7,  0.6,        0.8,        0.8,        0.6]
	
	# #  4.7Mb
	# prune_percentages =   [0,    0,          0,          0.5,        0.5,        0.5,
	#                        0.75, 0.75,       0.75,       0.875,      0.875]
	
	log_dir = './logs/002pruned_net_for_train/002-5.3/'
	cfg = []
	for i_cfg in range(11):
		layer_channels_i_cfg = conv_lyr_output_dim[i_cfg]
		prune_percentage = prune_percentages[i_cfg]
		prune_num = int(layer_channels_i_cfg) - int(layer_channels_i_cfg * prune_percentage)
		cfg.append(prune_num)

	opts = {
		"model_path": log_dir + 'ep360-loss23.708-val_loss26.419_0.944.h5',
		"anchors_path": 'model_data/tiny_yolo_mango_anchors.txt',
		"classes_path": 'model_data/mango_classes.txt',
		"score": 0.5*0.5,  # for object score and classification score respectively
		"iou": 0.5,
		"model_image_size": (512, 512),  # (416, 416)(512, 512)(608, 608)
		"gpu_num": 1,
		"max_boxes": 40,
		"cfg": cfg
	}
	# class YOLO defines the default value, so suppress any default here
	parser = argparse.ArgumentParser(
		argument_default=argparse.SUPPRESS)

	parser.add_argument(
		'--model_path', type=str, default=opts["model_path"],
		help='path to model weight file, default '
	)

	parser.add_argument(
		'--anchors_path', type=str, default=opts["anchors_path"],
		help='path to anchor definitions, default '
	)

	parser.add_argument(
		'--classes_path', type=str, default=opts["classes_path"],
		help='path to class definitions, default '
	)
	parser.add_argument(
		'--score', type=int, default=opts["score"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--iou', type=int, default=opts["iou"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--cfg', type=int, default=opts["cfg"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--model_image_size', type=int, default=opts["model_image_size"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--gpu_num', type=int, default=opts["gpu_num"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--max_boxes', type=int, default=opts["max_boxes"],
		help='Number of GPU to use, default '
	)
	parser.add_argument(
		'--image', default=True, action="store_true",
		help='Image detection mode, will ignore all positional arguments'
	)
	FLAGS = parser.parse_args()
	yolo = YOLO(**vars(FLAGS))
	detect_img(yolo)


if __name__ == '__main__':
	_main()
