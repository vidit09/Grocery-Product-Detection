#!/usr/bin/env python

# --------------------------------------------------------
#  Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

# --------------------------------------------------------
# Adapted by Vidit
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_path
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


CLASSES_FULL = ('__background__',
                 'Bakery','Biscuits','Candy/Bonbons',
                'Candy/Chocolate','Cereals','Chips',
                'Coffee','Dairy/Cheese','Dairy/Creme',
                'Dairy/Yoghurt','DriedFruitsAndNuts',
                'Drinks/Choco','Drinks/IceTea',
                'Drinks/Juices','Drinks/Milk',
                'Drinks/SoftDrinks','Drinks/Water',
                'Jars-Cans/Canned','Jars-Cans/Sauces',
                'Jars-Cans/Spices','Jars-Cans/Spreads',
                'Oil-Vinegar','Pasta','Rice',
                'Snacks','Soups','Tea')


def vis_detections(im, class_name, dets,im_path, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        #print('Total Proposed bbox:{}'.format(dets.shape[0]))
        #print('Total Proposed bbox after threshold:{}'.format(len(inds)))
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print("Detection Probability:{}".format(score))
        print("Class:{}".format(class_name))
        sub_mat = im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),2].copy().astype(int)
        sub_mat += 200
        sub_mat = (sub_mat*255/np.max(sub_mat)).astype(np.uint)
        im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), 1] = sub_mat

        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,0),3)
        cv2.putText(im,'{}:{:.2f}'.format(class_name,round(score,2)),(int(bbox[0]),int(bbox[1]+50)),cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,cv2.LINE_AA)

    path = os.path.dirname(im_path)
    im_name = os.path.basename(im_path)
    out_image = path+'/'+im_name.split('.')[0]+'_out.jpg'
    print('Saving output at {}'.format(out_image))
    cv2.imwrite(out_image,im)

def demo(sess, net, image_name, classes, thconf):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(image_name)
    im = cv2.imread(im_file)

    print(im.shape)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    print(scores.shape)
    print(len(classes))
    # Visualize detections for each class
    CONF_THRESH = thconf
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, im_file, thresh=CONF_THRESH)
#        break

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Grocery Product Faster R-CNN demo')
    parser.add_argument('--box', dest='box_type', help='Type of Bounding Boxes:[big,small]',
                         default='big')
    parser.add_argument('--img', dest='img',help='Image Path')
    parser.add_argument('--conf',dest='thconf',help='Confidence Threshold',type=float, default=0.5)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    box_type = args.box_type
    img = args.img
    thconf = args.thconf


    classes = CLASSES_FULL

    if box_type == 'big':
        tfmodel = os.path.join('output','big_box','big_box_weights.ckpt')
        cfg.TEST.SCALES[0] = 400
        cfg.TEST.MAX_SIZE  = 600
    else:
        tfmodel = os.path.join('output','small_box','small_box_weights.ckpt')
        cfg.TEST.SCALES[0] = 1000
        cfg.TEST.MAX_SIZE  = 1200

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\n').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network

    net = resnetv1(num_layers=101)

    net.create_architecture("TEST", len(classes),
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = [img]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_name))
        demo(sess, net, im_name, classes,thconf)


