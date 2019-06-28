# Pruned_CPUlevel_YOLO4Mango
For the manuscript 'Real-time mango detection on CPU using pruned YOLO network'

## Introduction
A Keras implementation of pruned YOLOv3-tiny to detect mango data (Tensorflow backend).
Original YOLO implement is inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

## How to use
1. Processing your data. The mango dataset is opensource can be found in [here](http://acquire.cqu.edu.au:8080/vital/access/manager/Repository/cqu:17570).
  Mango dataset labels are all xml files. So *voc_annotation.py* is for you to transfer them into txt file for our code.

2. The COCO2017 dataset can be found [here](http://cocodataset.org/#download). COCO website offer a lot of [API](https://github.com/cocodataset/cocoapi) for quickly using the COCO data. You can easily get the apple and orange images as you need.

3. Download YOLOv3-tiny pre-trained weigths from [YOLO website](http://pjreddie.com/darknet/yolo/). Then *convert.py* can be used to transfer the _.weights_ file to _.h5_ file.

4. Now the GradAM (gradient of target output with respect to the activation maps * activation maps) can be computed using *compute_grad_am.py*. We also upload the results in *model_data/grad_am_sort_idx_L.npy*, so you can directly use it to prune the YOLOv3-tiny.

5. Some description for code in main dir. 
   * *kmeans.py* is for computing anchor sizes. 
   * *compute_grad_am.py* is used for computing F1 scores of all network. 
   * *compute_flops.py* is used for computing FLOPs in Table 1 of the manuscript.

6. Then we will use the GradAM to prune the network. The pruning and training codes will be updated sooner. The trained weights and network structure can be find in *logs* for validate the results in our manuscript.

7. Some description for directories. 
   * Directory _data_annotation_ stores formatted txt file of mango dataset.
   * Directory _dataset_ stores COCO apple and orange images and mango dataset.
   * Directory _logs_ stores trained network weights.
   * Directory _yolo3_ contains all network building codes and some utils.

## Try it
1. For the Table 1 in the manuscript
    * After reformatting the mango data labels to _.txt_, run _detect_mango_cfg.py_ to generate the F1 score in Table 1. You can change param cfg to use other trained networks to detect the mango.
   
