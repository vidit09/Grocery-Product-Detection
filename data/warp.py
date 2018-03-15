#Wraping function
import numpy as np
import cv2
import os
import re

from helper import transform_bbox, find_rnd_bbox
from grid_pattern_helper import create_stacks_from_aug_image, create_stacks_from_single_image, create_stacks_from_mult_image

def wrap_fg_bg(im,im_fg_path,dst,stacked,aug=False):

    im_fg_mask = None

    if aug:
        _im_fg = []
        _fg_bbox = []
        n_row = np.random.randint(1, 3, 1)[0]
        n_col = np.random.randint(1, 4, 1)[0]
        for path in im_fg_path:
            fg = cv2.imread(path)
            seq = aug_transforms()
            au_im,au_bbox = create_stacks_from_aug_image(fg, seq, n_row, n_col)
            _im_fg.append(au_im)
            _fg_bbox.append(au_bbox)

    else:

        _im_fg = cv2.imread(im_fg_path)
        _im_name = os.path.basename(im_fg_path)
        im_dir_path = os.path.dirname(im_fg_path)

        for num in re.findall('\d+', _im_name):
            mask_path = im_dir_path + '/' + num + '_mask.jpg'
            im_fg_mask = cv2.imread(mask_path, 0)

    if aug:
        im_fg, mask_fg, _rects = create_stacks_from_mult_image(_im_fg, im)
        for id, au_bboxes in enumerate(_fg_bbox):
            for au_bbox in au_bboxes:
                au_bbox[0] += _rects[id][0]
                au_bbox[1] += _rects[id][1]
                au_bbox[2] += _rects[id][0]
                au_bbox[3] += _rects[id][1]

        rects = _fg_bbox

    else:
        im_fg, mask_fg, rects = create_stacks_from_single_image(_im_fg, im_fg_mask, stacked)

    row_fg, col_fg, _ = im_fg.shape

    src = np.array([[x, y] for x in [0, col_fg] for y in [0, row_fg]])

    if dst is None:
        if aug:
            aspect_ratio = im_fg.shape[0] / im_fg.shape[1]
        else:
            aspect_ratio = _im_fg.shape[0]/_im_fg.shape[1]
        rnd_bbox = find_rnd_bbox(im.shape,aspect_ratio)
        if rnd_bbox is None:
            return None,None,None
        dst = np.array([[x, y] for x in [rnd_bbox[0], rnd_bbox[2]] for y in [rnd_bbox[1], rnd_bbox[3]]])

    h, status = cv2.findHomography(src, dst)
    im_out = cv2.warpPerspective(im_fg, h, (im.shape[1], im.shape[0]))

    n_rects = []
    for rect_id in range(len(rects)):
        n_rects.append(transform_bbox(rects[rect_id], h))

    gen_bbox = []
    if not aug:
        x_min = min(dst[:, 0])
        y_min = min(dst[:, 1])
        x_max = max(dst[:, 0])
        y_max = max(dst[:, 1])
        gen_bbox = [x_min, y_min, x_max, y_max]
    else:
        for rect in _rects:
            gen_bbox.append(transform_bbox(rect, h))

    if mask_fg is not None:
        mask_fg_out = cv2.warpPerspective(mask_fg, h, (im.shape[1], im.shape[0]))
        alpha = mask_fg_out
    else:
        alpha = np.zeros((im.shape[0], im.shape[1]), dtype='uint8')

        alpha[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
        alpha = np.dstack([alpha.T]*3).swapaxes(0, 1).reshape(im.shape[0], im.shape[1], 3)

    kernel = np.ones((5, 5), np.float32) / 25
    alpha = cv2.filter2D(alpha.astype('float'), -1, kernel)

    temp = cv2.multiply(1 - alpha, im.astype('float'))

    im_out = cv2.multiply(alpha, im_out.astype('float'))
    im = cv2.add(temp, im_out).astype('uint8')

    return im, n_rects, gen_bbox
