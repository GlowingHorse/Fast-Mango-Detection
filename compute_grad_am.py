"""
Compute GradAM
"""

import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import colorsys

from yolo3.yolo_cfg import YOLO


def compute_grad_am(yolo):
	test_mango_img_path = './data_annotation/2007_test.txt'
	with open(test_mango_img_path) as f:
		test_lines = f.readlines()

	TPFN = 0.0
	TPFP = 0.0
	TP = 0.0

	hsv_tuples = [(1, 1., 1.)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
			colors))

	allimg_grad_ams = []
	for i_img_idx in range(len(test_lines)):
		img_annotation = test_lines[i_img_idx].split(" ")
		img_name = img_annotation[0]
		boxes = img_annotation[1:]
		true_box = []
		for box in boxes:
			true_box.append(np.array(list(map(int, box.split(',')))))


		image = Image.open(img_name)
		r_image, out_boxes, alllyr_grad_ams = yolo.compute_gradam_and_detect_box(image)
		allimg_grad_ams.append(alllyr_grad_ams)
		pix = np.array(r_image)

		fig102 = plt.figure(102)
		plt.imshow(pix)
		plt.show(block=False)

		for i_true_box in range(len(true_box)):
			left_true, top_true, right_true, bottom_true, class_label = true_box[i_true_box]
			for i_out_boxes in range(len(out_boxes)):
				top, left, bottom, right = out_boxes[i_out_boxes]
				center_x = abs(bottom-top)/2 + top
				center_y = abs(right - left) / 2 + left
				if top_true < center_x < bottom_true and left_true < center_y < right_true:
					TP = TP+1
					# remove used out_box
					out_boxes[i_out_boxes] = 0, 0, 0, 0

		TPFP = TPFP + len(out_boxes)
		TPFN = TPFN + len(true_box)
		print('TP : {}, TPFP :{}, TPFN :{} '.format(TP, TPFP, TPFN))
		print('\n')
	sum_grad_ams = 0
	for i_img_idx in range(len(allimg_grad_ams)):
		sum_grad_ams = sum_grad_ams + np.array(allimg_grad_ams[i_img_idx])

	sort_idx_L = []
	for i_lyr_num in range(len(sum_grad_ams)):
		sort_idx_L.append(np.argsort(sum_grad_ams[i_lyr_num]))

	np.save("sort_idx_L.npy", sort_idx_L)
	recall = TP / TPFN
	precision = TP / TPFP
	f1score = 2*recall*precision/(recall+precision)
	print("F1score : {}".format(f1score))
	# plt.close(fig101)
	plt.close(fig102)
	yolo.close_session()


FLAGS = None


if __name__ == '__main__':
	opts = {
		"model_path": './model_struc_and_ori_weight/yolov3-tiny.h5',
		"anchors_path": 'model_data/tiny_yolo_anchors.txt',
		"classes_path": 'model_data/coco_classes.txt',
		"score": 0.02,
		"iou": 0.1,
		"model_image_size": (416, 416),  # (512, 512)(416, 416)
		"gpu_num": 1,
		"max_boxes": 40
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

	compute_grad_am(YOLO(**vars(FLAGS)))
