# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers import Input
from yolo3.model import tiny_yolo_body

# install kerassurgeon first
from kerassurgeon.operations import delete_channels  # , delete_layer, insert_layer
# from kerassurgeon.identify import get_apoz
# from kerassurgeon import Surgeon
# from kerassurgeon import identify


def _main():
    log_dir = './logs/006MorePercent_Prune_net4_rain/001/'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)
    conv_lyr_idx = ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6",
                    "conv2d_7", "conv2d_8", "conv2d_11", "conv2d_9", "conv2d_12", "conv2d_10",
                    "conv2d_13"]
    conv_lyr_output_dim = [16,   32,        64,          128,        256,        512,
                           1024, 256,       128,         512,        256]

    # # 002 5.3Mb prune rates
    prune_percentages =   [0,    0,          0,          0.5,        0.5,        0.5,
                           0.75, 0.5,        0.75,       0.75,       0.75]

    sort_idx_L = np.load(log_dir+"grad_am_sort_idx_L.npy", allow_pickle=True)

    # prune the original yolo-v3-tiny
    model = create_tiny_model(input_shape, anchors, num_classes, log_dir + 'yolov3-tiny.h5')

    # total_channels = get_total_channels(model)
    conv_layer_names = []
    for layer in model.layers:
        if layer.__class__.__name__ == 'Conv2D':
            # if layer.output_shape[-1] > 32:
            conv_layer_names.append(layer.name)

    for i_selected_lyr in range(len(sort_idx_L) - 2):
        conv_layer_name = conv_layer_names[i_selected_lyr]
        conv_layers = model.get_layer(name=conv_layer_name)
        sort_idx = sort_idx_L[i_selected_lyr]
        prune_percentage = prune_percentages[i_selected_lyr]

        prune_num = int(len(sort_idx) * prune_percentage)
        prune_channels = sort_idx[:prune_num]
        prune_channels = prune_channels.tolist()
        model = delete_channels(model, conv_layers, prune_channels)

    save_pruned_net_name = log_dir + 'GradAM_pruned_yolo.h5'
    # Clean up tensorflow session after pruning and re-load model
    model.save_weights(save_pruned_net_name)

    del model
    K.clear_session()
    tf.reset_default_graph()


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


def create_tiny_model(input_shape, anchors, num_classes, weights_path=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w = input_shape
    # (416, 416) 32*multiplier

    num_anchors = len(anchors)
    model_body = tiny_yolo_body(Input(shape=(h, w, 3)), num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_body.load_weights(weights_path)

    return model_body


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


if __name__ == '__main__':
    _main()


