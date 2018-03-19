#Common Helper functions 
import numpy as np
import cv2

'''
Take smaller region out of second image
'''


def subset(mat1, mat2):

    if mat1.shape == mat2.shape:
        return mat2,mat2.shape[0],mat2.shape[1]
    else:
        min_row = min(mat1.shape[0], mat2.shape[0])
        min_col = min(mat1.shape[1], mat2.shape[1])

        return mat2[0:min_row, 0:min_col, :], min_row, min_col


'''
Resize image to make smaller side to target size and clamp to max size
'''


def resize(im, target_size, max_size):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
    return im_scale, im


'''
Change coordinates of bounding boxes using homography matrix(h)
'''


def transform_bbox(rects, h):

    rects = np.array(rects)
    if len(rects.shape) == 1:
        rects = rects[np.newaxis,:]
    rects = rects.reshape((rects.shape[0]*2,2)).T
    ones = np.ones((1,rects.shape[1]))

    rects = np.concatenate((rects,ones))

    # in final coordinates
    transform_rect = h.dot(rects)
    transform_rect[0,:] = transform_rect[0,:]/transform_rect[2,:]
    transform_rect[1,:] = transform_rect[1,:]/transform_rect[2,:]

    transform_rect = transform_rect[0:2,:].T
    transform_rect = transform_rect.reshape(int(transform_rect.shape[0]/2),4)

    # final transformed
    transform_rect = [[int(x) for x in rect] for rect in transform_rect]

    return transform_rect


'''
Sample random region with given aspect ratio
'''


def find_rnd_bbox(bg_im_shape,fg_aspect_ratio):

    W = bg_im_shape[1]
    H = bg_im_shape[0]

    # avoid boundary of image
    margin = 10
    x_min = np.random.randint(margin, W-margin, 1)[0]

    y_min = np.random.randint(margin, H-margin, 1)[0]

    # find height, width for 1/3rd area of actual image
    w = int(np.sqrt(W * H / (3 * fg_aspect_ratio)))
    h = int(fg_aspect_ratio * w)

    # find the valid x_min and y_min until counter expires
    counter = 100
    while counter > 0:

        exit_loop = 1

        # sample integer in the range s.t. still in background image height and weight
        if x_min+w > W:
            x_min = np.random.randint(margin, W - margin, 1)[0]
            exit_loop = 0

        if y_min+h > H:
            y_min = np.random.randint(margin, H - margin, 1)[0]
            exit_loop = 0
        counter -= 1

        if exit_loop:
            break

    if counter > 0:
        return [x_min, y_min, x_min+w, y_min+h]
    else:
        return None
