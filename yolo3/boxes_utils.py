from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_boxes_and_scores_ori_shape(feats, anchors, num_classes, input_shape, image_shape, score_threshold):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores_ori = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores_ori, [-1, num_classes])

    class_clear_idx = np.zeros((1, 80))
    class_clear_idx[0, 32] = 1
    class_clear_idx[0, 47] = 1
    class_clear_idx[0, 49] = 1
    class_clear_idx = tf.convert_to_tensor(class_clear_idx, dtype=tf.float32)
    # class_clear_idx = class_clear_idx.astype(bool)
    box_scores_ori = tf.multiply(box_scores_ori, class_clear_idx)

    mask_scores_ori = box_scores_ori[..., 32] + box_scores_ori[..., 47] + box_scores_ori[..., 49]
    mask_scores_ori = mask_scores_ori > score_threshold
    mask_scores_ori = tf.expand_dims(tf.cast(mask_scores_ori, dtype=tf.float32), axis=len(mask_scores_ori.shape))
    box_xy_masked = box_xy * mask_scores_ori
    box_wh_masked = box_wh * mask_scores_ori
    box_confidence_masked = box_confidence * mask_scores_ori

    output_scores_ori = tf.concat([box_xy_masked, box_wh_masked, box_confidence_masked, box_scores_ori], axis=-1)
    shape_temp = tf.shape(output_scores_ori)
    # output_scores_ori = K.reshape(output_scores_ori, [shape_temp[0], shape_temp[1], shape_temp[2], 255])

    output_scores_ori = K.reshape(output_scores_ori, [-1, shape_temp[1], shape_temp[2], 255])
    # output_scores_ori = K.expand_dims(output_scores_ori, axis=0)

    # want_boxes_L = [box_scores_ori, mask_scores_ori]
    return boxes, box_scores, output_scores_ori  #  , want_boxes_L

