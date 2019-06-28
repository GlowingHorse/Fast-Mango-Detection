"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
from keras import backend as K
from re import findall


def save_imgs(images, save_directory, attr_class, factorization_method
			  , no_slash_layer_name, imgtype_name='opt'):
	for i_optimgs in range(len(images)):
		if len(images[i_optimgs]) > 1:
			images_temp = images[i_optimgs]
			w = int(np.sqrt(images_temp.size / 3))
			img = images_temp.reshape(w, w, 3)
			factorization_method = findall('[A-Z]', factorization_method)
			factorization_method = ''.join(factorization_method)
			plt.imsave(save_directory + "/" + attr_class + '_' + factorization_method + '_' +
					   no_slash_layer_name + '_' + imgtype_name + str(i_optimgs) + ".jpg", img)


def plot_gradam(data, save_directory, conv_lyr_idx, lyr_channel_idx, save_img_name,
				xi=None, cmap='RdBu_r', alpha=0.8):
	plt.ioff()
	fig = plt.figure(1, figsize=[4.16, 4.16], dpi=200, frameon=False)

	axis = plt.Axes(fig, [0., 0., 1., 1.])
	axis.set_axis_off()
	fig.add_axes(axis)

	dx, dy = 0.05, 0.05
	xx = np.arange(0.0, data.shape[1] + dx, dx)
	yy = np.arange(0.0, data.shape[0] + dy, dy)
	xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
	extent = xmin, xmax, ymin, ymax
	cmap_xi = plt.get_cmap('jet')
	cmap_xi.set_bad(alpha=0)
	overlay = xi
	if len(data.shape) == 3:
		data = np.mean(data, 2)
	axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
	axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
	# plt.show()
	plt.savefig(save_directory + '/' + 'gam_' + str(conv_lyr_idx) + '_' + str(
		lyr_channel_idx) + save_img_name)  # 'RdBu_r' 'hot'
	plt.close(1)


def compose(*funcs):
	"""Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
	# return lambda x: reduce(lambda v, f: f(v), funcs, x)
	if funcs:
		return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
	else:
		raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
	'''resize image with unchanged aspect ratio using padding'''
	iw, ih = image.size
	w, h = size
	scale = min(w / iw, h / ih)
	nw = int(iw * scale)
	nh = int(ih * scale)

	image = image.resize((nw, nh), Image.BICUBIC)
	new_image = Image.new('RGB', size, (128, 128, 128))
	new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
	return new_image


def rand(a=0, b=1):
	return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=40, jitter=.3, hue=.1, sat=1.5, val=1.5,
					proc_img=True):
	'''random preprocessing for real-time data augmentation'''
	line = annotation_line.split()
	image = Image.open(line[0])
	iw, ih = image.size
	h, w = input_shape
	box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

	if not random:
		# resize image
		scale = min(w / iw, h / ih)
		nw = int(iw * scale)
		nh = int(ih * scale)
		dx = (w - nw) // 2
		dy = (h - nh) // 2
		image_data = 0
		if proc_img:
			image = image.resize((nw, nh), Image.BICUBIC)
			new_image = Image.new('RGB', (w, h), (128, 128, 128))
			new_image.paste(image, (dx, dy))
			image_data = np.array(new_image) / 255.

		# correct boxes
		box_data = np.zeros((max_boxes, 5))
		if len(box) > 0:
			np.random.shuffle(box)
			if len(box) > max_boxes: box = box[:max_boxes]
			box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
			box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
			box_data[:len(box)] = box

		draw = ImageDraw.Draw(new_image)

		for i_true_box in range(len(box_data)):
			left_true, top_true, right_true, bottom_true, class_label = box_data[i_true_box]
			for i in range(3):
				draw.rectangle([left_true + i, top_true + i, right_true - i, bottom_true - i])
		del draw
		# new_image.show()
		# pix = np.array(new_image)

		# fig102 = plt.figure(102)
		# plt.imshow(pix)
		# plt.show(block=False)
		# plt.close(fig102)

		return image_data, box_data

	# resize image
	new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
	scale = rand(.25, 2)
	if new_ar < 1:
		nh = int(scale * h)
		nw = int(nh * new_ar)
	else:
		nw = int(scale * w)
		nh = int(nw / new_ar)
	image = image.resize((nw, nh), Image.BICUBIC)

	# place image
	dx = int(rand(0, w - nw))
	dy = int(rand(0, h - nh))
	new_image = Image.new('RGB', (w, h), (128, 128, 128))
	new_image.paste(image, (dx, dy))
	image = new_image

	# flip image or not
	flip = rand() < .5
	if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

	# distort image
	hue = rand(-hue, hue)
	sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
	val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
	x = rgb_to_hsv(np.array(image) / 255.)
	x[..., 0] += hue
	x[..., 0][x[..., 0] > 1] -= 1
	x[..., 0][x[..., 0] < 0] += 1
	x[..., 1] *= sat
	x[..., 2] *= val
	x[x > 1] = 1
	x[x < 0] = 0
	image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

	# correct boxes
	box_data = np.zeros((max_boxes, 5))
	if len(box) > 0:
		np.random.shuffle(box)
		box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
		box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
		if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
		box[:, 0:2][box[:, 0:2] < 0] = 0
		box[:, 2][box[:, 2] > w] = w
		box[:, 3][box[:, 3] > h] = h
		box_w = box[:, 2] - box[:, 0]
		box_h = box[:, 3] - box[:, 1]
		box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
		if len(box) > max_boxes: box = box[:max_boxes]
		box_data[:len(box)] = box

	return image_data, box_data
