# Grid Pattern Helper Function

import numpy as np
import cv2

from helper import subset

'''
Create Grid like pattern from single images obtained from application of transforms
'''


def create_stacks_from_aug_image(im, seq, n_row, n_col):

    # get the max width and height in image list
    aug_ims = []
    r_max = -np.inf
    c_max = -np.inf

    for augs in range(n_row*n_col):
        aug_im = seq.augment_images([im])[0]
        r_max = max(r_max, aug_im.shape[0])
        c_max = max(c_max, aug_im.shape[1])
        aug_ims.append(aug_im)

    # add borders to the smaller images to make all of same size
    for id in range(len(aug_ims)):
        r_diff = r_max-aug_ims[id].shape[0]
        c_diff = c_max-aug_ims[id].shape[1]
        aug_ims[id] = cv2.copyMakeBorder(aug_ims[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)

    # reshape the to get grid structure
    aug_ims = np.array(aug_ims)
    aug_ims = np.reshape(aug_ims,(n_row,n_col,r_max,c_max,3)).swapaxes(1,2)
    aug_ims = np.reshape(aug_ims,(r_max*n_row,c_max*n_col,3))

    # all the small boxes
    bbox = []
    for r in range(n_row):
        for c in range(n_col):
            bb = []
            bb.append(c*c_max)
            bb.append(r*r_max)
            bb.append((c+1)*c_max)
            bb.append((r+1)*r_max)
            bbox.append(bb)

    return aug_ims,bbox


'''
Create Grid like pattern from single images and mask of Grocery Product
'''


def create_stacks_from_single_image(im, mask, stacked):

    # Random sample row and col size
    if stacked:
        n_row = np.random.randint(1,3,1)[0]
        n_col = np.random.randint(1,3,1)[0]
    else:
        n_row = 1
        n_col = 1

    # final stacked images and mask
    n_im = np.zeros((im.shape[0]*n_row,im.shape[1]*n_col,3));
    n_mask = np.zeros((im.shape[0]*n_row,im.shape[1]*n_col,3));

    rects = []
    temp = mask.copy()
    for r in range(n_row):
        for c in range(n_col):

            _im = im

            if mask is not None:

                # convert mask to binary
                mask[temp >= 125] = 1
                mask[temp < 125] = 0

                # mask from single channel to three channel
                maskc = np.dstack([mask.T] * 3).swapaxes(0, 1).reshape(mask.shape[0], mask.shape[1], 3)

                # get only the product region
                imc = cv2.multiply(_im.astype('uint8'), maskc.astype('uint8'))

                # get the location where product is
                y = np.where(maskc == 1)[0]
                x = np.where(maskc == 1)[1]

                # bounding box around the product in the image
                x_min = np.min(x)
                x_max = np.max(x)

                y_min = np.min(y)
                y_max = np.max(y)

                # place the product in the stacks
                mat,row,col = subset(_im,imc)
                n_im[r*_im.shape[0]:r*_im.shape[0]+row,c*_im.shape[1]:c*_im.shape[1]+col,:] = mat

                mat, row, col = subset(mask, maskc)
                n_mask[r*mask.shape[0]:r*mask.shape[0]+row,c*mask.shape[1]:c*mask.shape[1]+col,:] = mat

            else:

                # image in entirety taken as mask
                n_im[r * _im.shape[0]:(r+1) * _im.shape[0], c * _im.shape[1]:(c+1) * _im.shape[1], :] = im
                n_mask = None
                x_min = 0
                y_min = 0
                x_max = im.shape[1]
                y_max = im.shape[0]

            # final smaller bounding boxes
            y_ = im.shape[0]
            x_ = im.shape[1]
            bbox=[]
            bbox.append(x_min+c*x_)
            bbox.append(y_min+r*y_)
            bbox.append(x_max+c*x_)
            bbox.append(y_max+r*y_)

            rects.append(bbox)

    if n_mask is not None:
        n_mask.astype('uint8')
    return n_im, n_mask, rects


'''
Create Grid like pattern with multiple images
'''


def create_stacks_from_mult_image(im_list,bg_im):

    bg_im_shape = bg_im.shape

    num_img = len(im_list)
    masks = []
    gt_bbox = []

    # get the max width and height in image list
    r_max = -np.inf
    c_max = -np.inf

    for im in im_list:
        r_max = max(r_max, im.shape[0])
        c_max = max(c_max, im.shape[1])
        masks.append(np.ones(im.shape))
        gt_bbox.append([0, 0, im.shape[1], im.shape[0]])

    # add borders to the smaller images to make all of same size
    for id in range(len(im_list)):
        if bg_im_shape[0] >= bg_im_shape[1]:
            r_diff = 0
            c_diff = c_max-im_list[id].shape[1]

        else:
            r_diff = r_max-im_list[id].shape[0]
            c_diff = 0

        im_list[id] = cv2.copyMakeBorder(im_list[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)
        masks[id] = cv2.copyMakeBorder(masks[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)

    # stack the images depending on longer side
    if bg_im_shape[0] >= bg_im_shape[1]:
        r=0
        for id in range(num_img):
           r += im_list[id].shape[0]
        ims = np.zeros((r,c_max,3))
        aug_masks = np.zeros((r,c_max,3))
        # list of small boxes
        for id in range(1, len(gt_bbox)):
            gt_bbox_prev = gt_bbox[id-1]
            gt_bbox[id][1] = gt_bbox_prev[3]
            gt_bbox[id][3] = gt_bbox[id][3]+gt_bbox_prev[3]

        for id in range(num_img):
            ims[gt_bbox[id][1]:gt_bbox[id][3], 0:c_max, :] = im_list[id]
            aug_masks[gt_bbox[id][1]:gt_bbox[id][3], 0:c_max, :] = masks[id]
    else:
        c = 0
        for id in range(num_img):
            c += im_list[id].shape[1]
        ims = np.zeros((r_max,c,3))
        aug_masks = np.zeros((r_max,c,3))

        # list of small boxes
        for id in range(1, len(gt_bbox)):
            gt_bbox_prev = gt_bbox[id - 1]
            gt_bbox[id][0] = gt_bbox_prev[2]
            gt_bbox[id][2] = gt_bbox[id][2] + gt_bbox_prev[2]

        for id in range(num_img):
            ims[0:r_max, gt_bbox[id][0]:gt_bbox[id][2], :] = im_list[id]
            aug_masks[0:r_max, gt_bbox[id][0]:gt_bbox[id][2], :] = masks[id]

    return ims,aug_masks,gt_bbox
