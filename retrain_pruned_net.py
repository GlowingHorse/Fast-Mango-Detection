"""
Finetuning the YOLO model for your own dataset.
"""

import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_loss, tiny_yolo_body_with_cfg
from yolo3.utils import get_random_data
from keras.utils.vis_utils import plot_model


def _main():
    train_annotation_path = './data_annotation/train.txt'
    val_annotation_path = './data_annotation/test.txt'
    log_dir = './logs/002pruned_net_for_train/003/'
    classes_path = 'model_data/mango_classes.txt'
    anchors_path = 'model_data/tiny_yolo_mango_anchors.txt'
    # weights_path = log_dir + 'GradAM_pruned_yolo.h5'
    # sort_idx_L = np.load(log_dir + "grad_am_sort_idx_L.npy", allow_pickle=True)

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (512, 512)  # multiple of 32, hw
    conv_lyr_idx = ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6",
                    "conv2d_7", "conv2d_8", "conv2d_11", "conv2d_9", "conv2d_12"]
    conv_lyr_output_dim = [16,   32,        64,          128,        256,        512,
                           1024, 256,       128,         512,        256]

    #  003 5.2Mb
    prune_percentages =   [0.2,    0,          0,          0.5,        0.5,        0.5,
                           0.75,   0.6,        0.75,       0.75,       0.75]

    # # 002 5.3Mb
    # prune_percentages =   [0,    0,          0,          0.5,        0.5,        0.5,
    #                        0.75, 0.5,        0.75,       0.75,       0.75]

    # #  6.3Mb
    # prune_percentages =   [0,    0,          0,          0.4,        0.4,        0.5,
    #                        0.7,  0.6,        0.8,        0.8,        0.6]
    #
    # #  4.7Mb
    # prune_percentages =   [0,    0,          0,          0.5,        0.5,        0.5,
    #                        0.75, 0.75,       0.75,       0.875,      0.875]

    cfg = []
    for i_cfg in range(11):
        layer_channels_i_cfg = conv_lyr_output_dim[i_cfg]
        prune_percentage = prune_percentages[i_cfg]
        prune_num = int(layer_channels_i_cfg) - int(layer_channels_i_cfg * prune_percentage)
        cfg.append(prune_num)

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_pruned_tiny_model_with_cfg(input_shape, anchors, num_classes, load_pretrained=False,
                              cfg=cfg, freeze_body=2)  # , weights_path=weights_path)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=10)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=45, verbose=1, min_lr=1e-7)
    reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, verbose=1, min_lr=1e-6)
    reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=35, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=120, verbose=1)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    # np.random.seed(None)
    num_val = len(val_lines)
    num_train = len(train_lines)

    epoch_warmup = 1500
    epoch_1 = 100
    epoch_2 = 600
    # epoch_3 = 800

    random_1_3 = True
    random_2_4 = False
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        print('Unfreeze all of the layers.')

        model.compile(optimizer=Adam(lr=1e-2), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
                data_generator_wrapper(
                    train_lines, batch_size, input_shape, anchors, num_classes, random=random_1_3),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(
                    val_lines, batch_size, input_shape, anchors, num_classes, random=random_2_4),
                validation_steps=max(1, num_val//batch_size),
                epochs=epoch_warmup,
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr1])

    # Further training if needed.
    # # plot_model(model, to_file='pruned_yolov3tiny_model_6.3.png', show_shapes=True, dpi=300)
    # if True:
    #     model.compile(optimizer=Adam(lr=1e-3),
    #                   loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
    #
    #     model.fit_generator(
    #         data_generator_wrapper(
    #             train_lines, batch_size, input_shape, anchors, num_classes, random=random_2_4),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(
    #             val_lines, batch_size, input_shape, anchors, num_classes, random=random_2_4),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=epoch_1,
    #         initial_epoch=epoch_warmup,
    #         callbacks=[logging, checkpoint, reduce_lr1])
    #
    # if True:
    #     model.compile(optimizer=Adam(lr=1e-3),
    #                   loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
    #
    #     model.fit_generator(
    #         data_generator_wrapper(
    #             train_lines, batch_size, input_shape, anchors, num_classes, random=random_1_3),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(
    #             val_lines, batch_size, input_shape, anchors, num_classes, random=random_2_4),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=epoch_2,
    #         initial_epoch=epoch_1,
    #         callbacks=[logging, checkpoint, reduce_lr2])

        model.save_weights(log_dir + 'trained_weights_final.h5')


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


def create_pruned_tiny_model_with_cfg(input_shape, anchors, num_classes, load_pretrained=True,
                                      cfg=None, freeze_body=2, weights_path=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w = input_shape

    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body_with_cfg(image_input, num_anchors//2, num_classes, cfg)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]

            freeze_lyr_name = []
            for i in range(num):
                freeze_lyr_name.append(model_body.layers[i].name)
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)


if __name__ == '__main__':
    _main()
