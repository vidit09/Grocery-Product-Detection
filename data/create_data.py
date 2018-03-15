import cv2
import numpy as np
from imgaug import augmenters as iaa
import os,glob
import re
import yaml

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir,'augmentation_param.yml'), 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

print (cfg)

NUM_SHELF_BG_SAMPLE = cfg['NUM_SHELF_BG_SAMPLE']
NUM_NEG_BG_SAMPLE = 2
TARGET_SIZE = 600
MAX_SIZE = 1000

out_dir = 'data2/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_img_dir = out_dir+'Images/'
if not os.path.exists(out_img_dir):
    os.mkdir(out_img_dir)

product_img_dir = '/Users/vidit/Thesis/Grocery_products/'
product_img_path = product_img_dir+'TrainingFiles.txt'
product_cat_exp = 'Training/Food'

cat_map = []
with open(product_img_dir+'cat_mapping.txt') as f:
    for line in f:
        cat_map.append(product_img_dir+line.strip())

with open(product_img_path) as f:
    train_files = f.readlines()

products = [p for p in filter(lambda x: product_cat_exp in x, train_files)]
product_files = []

with open(out_dir+'gt_mapping.txt','w') as f:
    for product in products:
        f.write(product)
        product_files.append(product_img_dir + product)



bg_dir = '/Users/vidit/Thesis/shelf_images/'
bg_annotation = bg_dir + 'Annotations/'
bg_annotation = glob.glob(bg_annotation+'*.txt')

bg_annotation_grouped = bg_dir + 'Annotations_Grouped/'
bg_annotation_grouped = glob.glob(bg_annotation_grouped+'*.txt')

bg_img = bg_dir + 'Images_all/'
bg_img = glob.glob(bg_img+'*.jpg')

neg_bg_path1 = '/Users/vidit/Thesis/Grocery_products/Training/Background/'
neg_bg_img = glob.glob(neg_bg_path1+'*.png')

neg_bg_path2 = '/Users/vidit/Thesis/coco/images/train2014/'
neg_bg_img2 = glob.glob(neg_bg_path2+'*.jpg')

neg_bg_img.extend(neg_bg_img2)

aug_annotation_file = open(out_dir+'annotations.txt','w')
aug_annotation_file_with_cat = open(out_dir+'annotations_with_big_box_cat.txt','w')
aug_annotation_file_small_boxes = open(out_dir+'annotations_with_small_box_cat.txt','w')


def aug_transforms():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                shear=(-6, 6),  # shear by -16 to +16 degrees
            )),

            iaa.SomeOf((0, 5),
                       [
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images

                           iaa.Add((-10, 10)),
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5)),

                           ]),
                           iaa.ContrastNormalization((0.1, 2.0)),  # improve or worsen the contrast
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    return seq


def subset(mat1,mat2):
    if mat1.shape == mat2.shape:
        return mat2,mat2.shape[0],mat2.shape[1]
    else:
        min_row = min(mat1.shape[0], mat2.shape[0])
        min_col = min(mat1.shape[1], mat2.shape[1])

        return mat2[0:min_row,0:min_col,:],min_row,min_col


def resize(im):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(TARGET_SIZE) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = float(MAX_SIZE) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
    return im_scale, im


def transform_bbox(rects,h):
    rects = np.array(rects)
    if len(rects.shape) == 1:
        rects = rects[np.newaxis,:]
    rects = rects.reshape((rects.shape[0]*2,2)).T
    ones = np.ones((1,rects.shape[1]))

    rects = np.concatenate((rects,ones))
    transform_rect = h.dot(rects)
    transform_rect[0,:] = transform_rect[0,:]/transform_rect[2,:]
    transform_rect[1,:] = transform_rect[1,:]/transform_rect[2,:]

    transform_rect = transform_rect[0:2,:].T
    transform_rect = transform_rect.reshape(int(transform_rect.shape[0]/2),4)

    transform_rect = [[int(x) for x in rect] for rect in transform_rect]

    return transform_rect


def find_rnd_bbox(bg_im_shape,fg_aspect_ratio):

    W = bg_im_shape[1]
    H = bg_im_shape[0]

    margin = 10
    counter = 100
    x_min = np.random.randint(margin, W-margin, 1)[0]

    y_min = np.random.randint(margin, H-margin, 1)[0]

    w = int(np.sqrt(W * H / (3 * fg_aspect_ratio)))
    h = int(fg_aspect_ratio * w)

    while counter > 0:

        exit_loop = 1
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


def create_stacks_from_aug_image(im, seq, n_row, n_col):

    aug_ims = []
    r_max = -np.inf
    c_max = -np.inf

    for augs in range(n_row*n_col):
        aug_im = seq.augment_images([im])[0]
        r_max = max(r_max, aug_im.shape[0])
        c_max = max(c_max, aug_im.shape[1])
        aug_ims.append(aug_im)

    for id in range(len(aug_ims)):
        r_diff = r_max-aug_ims[id].shape[0]
        c_diff = c_max-aug_ims[id].shape[1]
        aug_ims[id] = cv2.copyMakeBorder(aug_ims[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)

    aug_ims = np.array(aug_ims)
    aug_ims = np.reshape(aug_ims,(n_row,n_col,r_max,c_max,3)).swapaxes(1,2)
    aug_ims = np.reshape(aug_ims,(r_max*n_row,c_max*n_col,3))

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


def create_stacks_from_single_image(im, mask, stacked):


    if stacked:
        n_row = np.random.randint(1,3,1)[0]
        n_col = np.random.randint(1,3,1)[0]
    else:
        n_row = 1
        n_col = 1

    n_im = np.zeros((im.shape[0]*n_row,im.shape[1]*n_col,3));
    n_mask = np.zeros((im.shape[0]*n_row,im.shape[1]*n_col,3));

    rects = []
    temp = mask.copy()
    for r in range(n_row):
        for c in range(n_col):

            # _im = images[r*n_row+c]
            _im = im
            # mask = get_mask(_im)

            if mask is not None:

                mask[temp >= 125] = 1
                mask[temp < 125] = 0

                maskc = np.dstack([mask.T] * 3).swapaxes(0, 1).reshape(mask.shape[0], mask.shape[1], 3)

                imc = cv2.multiply(_im.astype('uint8'), maskc.astype('uint8'))

                y = np.where(maskc == 1)[0]
                x = np.where(maskc == 1)[1]

                x_min = np.min(x)
                x_max = np.max(x)

                y_min = np.min(y)
                y_max = np.max(y)

                mat,row,col = subset(_im,imc)
                n_im[r*_im.shape[0]:r*_im.shape[0]+row,c*_im.shape[1]:c*_im.shape[1]+col,:] = mat

                mat, row, col = subset(mask, maskc)
                n_mask[r*mask.shape[0]:r*mask.shape[0]+row,c*mask.shape[1]:c*mask.shape[1]+col,:] = mat

            else:
                n_im[r * _im.shape[0]:(r+1) * _im.shape[0], c * _im.shape[1]:(c+1) * _im.shape[1], :] = im
                n_mask = None
                x_min = 0
                y_min = 0
                x_max = im.shape[1]
                y_max = im.shape[0]

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


def create_stacks_from_mult_image(im_list,bg_im):
    r_max = -np.inf
    c_max = -np.inf

    bg_im_shape = bg_im.shape

    num_img = len(im_list)
    masks = []
    gt_bbox = []

    for im in im_list:
        r_max = max(r_max, im.shape[0])
        c_max = max(c_max, im.shape[1])
        masks.append(np.ones(im.shape))
        gt_bbox.append([0, 0, im.shape[1], im.shape[0]])

    for id in range(len(im_list)):
        if bg_im_shape[0] >= bg_im_shape[1]:
            r_diff = 0
            c_diff = c_max-im_list[id].shape[1]

        else:
            r_diff = r_max-im_list[id].shape[0]
            c_diff = 0

        im_list[id] = cv2.copyMakeBorder(im_list[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)
        masks[id] = cv2.copyMakeBorder(masks[id], 0, r_diff, 0, c_diff, cv2.BORDER_CONSTANT, value=0)

    if bg_im_shape[0] >= bg_im_shape[1]:
        r=0
        for id in range(num_img):
           r += im_list[id].shape[0]
        ims = np.zeros((r,c_max,3))
        aug_masks = np.zeros((r,c_max,3))
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

        for id in range(1, len(gt_bbox)):
            gt_bbox_prev = gt_bbox[id - 1]
            gt_bbox[id][0] = gt_bbox_prev[2]
            gt_bbox[id][2] = gt_bbox[id][2] + gt_bbox_prev[2]



        for id in range(num_img):

            ims[0:r_max, gt_bbox[id][0]:gt_bbox[id][2], :] = im_list[id]
            aug_masks[0:r_max, gt_bbox[id][0]:gt_bbox[id][2], :] = masks[id]


    return ims,aug_masks,gt_bbox



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


def with_shelf_bg(offset, fg_ids, grouped=False, stacked=False):

    if grouped:
        annotation = bg_annotation_grouped
    else:
        annotation = bg_annotation

    bg_ids = np.random.choice(len(annotation), NUM_SHELF_BG_SAMPLE, replace=False)
    anno_sub = [annotation[bg_id] for bg_id in bg_ids]
    bg_img_sub = [bg_img[bg_id] for bg_id in bg_ids]
    for idx, anno_f in enumerate(anno_sub):

        with open(anno_f) as anno_file:
            annos = anno_file.readlines()

        np.random.shuffle(annos)

        im_bg = cv2.imread(bg_img_sub[idx])

        im_scale, im_bg = resize(im_bg)
        im = im_bg
        row, col, ch = im_bg.shape

        print('Processing: {}'.format(anno_f))

        im_name = 'im_{num:05}.jpg'.format(num=offset + idx)
        aug_annotation_file.write('#\n')
        aug_annotation_file.write(im_name + '\n')
        aug_annotation_file_with_cat.write('#\n')
        aug_annotation_file_with_cat.write(im_name + '\n')

        aug_annotation_file_small_boxes.write('#\n')
        aug_annotation_file_small_boxes.write(im_name + '\n')
        # for cnt, anno in enumerate(annos[:len(fg_ids)]):
        for cnt, anno in enumerate(annos):
            bbox = np.array([float(cor) for cor in anno.strip().split()])
            bbox[0] = bbox[0] * im_scale
            # bbox[1], bbox[2] = bbox[2], bbox[1]

            bbox[1] = bbox[1] * im_scale
            bbox[2] = bbox[2] * im_scale

            bbox[3] = bbox[3] * im_scale

            jitter1 = np.random.randint(-5, 5, 1)
            jitter2 = np.random.randint(-5, 5, 1)

            bbox[0] = min(max(int(bbox[0]) + jitter1, 0), col - 1)
            bbox[1] = min(max(int(bbox[1]) + jitter2, 0), row - 1)
            bbox[2] = min(max(int(bbox[2]) + jitter1, 0), col - 1)
            bbox[3] = min(max(int(bbox[3]) + jitter2, 0), row - 1)

            dst = np.array([[int(x), int(y)] for x in bbox[0::2] for y in bbox[1::2]])


            # im_fg_path,_ = product_files[fg_ids[cnt%len(fg_ids)]].strip().split('.')
            im_fg_path,_ = fg_ids[cnt%len(fg_ids)].strip().split('.')
            bkg_reduced_file = im_fg_path + '_bkg_reduced.jpg'
            im, n_rects, _ = wrap_fg_bg(im, bkg_reduced_file, dst, stacked)

            for rect_id, rects in enumerate(n_rects):
                for rect in rects:
                    im_fg_cat = os.path.dirname(im_fg_path)
                    id = cat_map.index(im_fg_cat)+1
                    aug_annotation_file_small_boxes.write(str(int(rect[0])) + ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(int(rect[3])) + ' '+ str(id)+'\n')




            # if BOX_BIG:
            #     n_rects = [[bbox]]

            n_rects = [[bbox]]

            for rect_id, rects in enumerate(n_rects):
                for rect in rects:
                    aug_annotation_file.write(str(int(rect[0]))+ ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(int(rect[3])) + '\n')

                    im_fg_cat = os.path.dirname(im_fg_path)
                    id = cat_map.index(im_fg_cat)+1
                    aug_annotation_file_with_cat.write(str(int(rect[0]))+ ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(int(rect[3])) + ' '+ str(id)+'\n')

                    # cv2.rectangle(im, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 3)

        cv2.imwrite(out_img_dir + im_name, im)


def with_neg_bg_bg_sampled(start_num, fg_ids, stacked=False):

    im_fg_path = []
    for id in fg_ids:
        dirname = os.path.dirname(id.strip())
        filename,_ = os.path.basename(id.strip()).split('.')
        bkg_reduced_file = os.path.join(dirname,filename+'_bkg_reduced.jpg')
        im_fg_path.append(bkg_reduced_file)
        print('Processing for image:{}'.format(bkg_reduced_file))

    neg_bg_ids = np.random.choice(len(neg_bg_img), NUM_NEG_BG_SAMPLE, replace=False)
    for idx, bg_id in enumerate(neg_bg_ids):

        im_bg = cv2.imread(neg_bg_img[bg_id])
        row, col, ch = im_bg.shape
        im = im_bg.copy()

        im, n_rects, bbox = wrap_fg_bg(im, im_fg_path, None, stacked,aug=True)

        if n_rects is not None:
            im_name = 'im_{num:05}.jpg'.format(num=idx + start_num)
            aug_annotation_file.write('#\n')
            aug_annotation_file.write(im_name + '\n')
            aug_annotation_file_with_cat.write('#\n')
            aug_annotation_file_with_cat.write(im_name + '\n')
            aug_annotation_file_small_boxes.write('#\n')
            aug_annotation_file_small_boxes.write(im_name + '\n')

            for rect_id,rects in enumerate(n_rects):
                for rect in rects:
                    im_fg_cat = os.path.dirname(im_fg_path[rect_id])
                    id = cat_map.index(im_fg_cat) + 1
                    aug_annotation_file_small_boxes.write(
                        str(int(rect[0])) + ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(
                            int(rect[3])) + ' ' + str(id) + '\n')

            # if BOX_BIG:
            #     n_rects = bbox

            n_rects = bbox
            for rect_id,rects in enumerate(n_rects):
                for rect in rects:
                    aug_annotation_file.write(
                        str(int(rect[0])) + ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(int(rect[3])) + '\n')

                    im_fg_cat = os.path.dirname(im_fg_path[rect_id])
                    id = cat_map.index(im_fg_cat) + 1
                    aug_annotation_file_with_cat.write(
                        str(int(rect[0])) + ' ' + str(int(rect[1])) + ' ' + str(int(rect[2])) + ' ' + str(
                            int(rect[3])) + ' ' + str(id) + '\n')
                    # cv2.rectangle(im, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 3)
            cv2.imwrite(out_img_dir + im_name, im)


product_files_per_class = [[p for p in product_files if cat in p] for cat in cat_map]

min_prod = np.inf
max_prod = -np.inf

for prod_per_class  in product_files_per_class:
    if min_prod > len(prod_per_class):
        min_prod = len(prod_per_class)

    if max_prod < len(prod_per_class):
        max_prod = len(prod_per_class)

offset_val = 0
for _ in range(3):
    batches = [[np.random.choice(prod_per_class,min_prod,replace=False) for _ in range(max_prod//min_prod)] for prod_per_class in product_files_per_class]

    batches = [item for batch_per_cat in batches for batch in batch_per_cat for item in batch]
    np.random.shuffle(batches)
    sample = 10
    for augs in range(len(batches)//sample):

        fgs = batches[augs*sample:(augs+1)*sample]
        with_shelf_bg(augs*NUM_SHELF_BG_SAMPLE+offset_val,fgs,grouped=True,stacked=True)

    fgs = batches[(augs+1)*sample:]
    if len(fgs) > 0:
        with_shelf_bg((augs+1) * NUM_SHELF_BG_SAMPLE + offset_val, fgs, grouped=True, stacked=True)
        augs = augs + 1

    offset_val += (augs+1) * NUM_SHELF_BG_SAMPLE

    sample = 2
    for augs in range(len(batches) // sample):
        fgs = batches[augs * sample:(augs + 1) * sample]
        with_neg_bg_bg_sampled(augs * NUM_NEG_BG_SAMPLE + offset_val, fgs, stacked=True)

    fgs = batches[(augs + 1) * sample:]
    if len(fgs) > 0:
        with_neg_bg_bg_sampled(augs * NUM_NEG_BG_SAMPLE + offset_val, fgs, stacked=True)
        augs = augs + 1

    offset_val += (augs+1) * NUM_NEG_BG_SAMPLE



# for _ in range(3):
#     batches = [[np.random.choice(prod_per_class,min_prod,replace=False) for _ in range(max_prod//min_prod)] for prod_per_class in product_files_per_class]
#
#
#     for id_class in range(len(cat_map)):
#         for iter in range(max_prod//min_prod):
#             batch = batches[id_class][iter]
#             augs = len(batch)//10
#             for id in range(augs):
#                 fgs = batch[id*10:(id+1)*10]
#                 with_shelf_bg(id*NUM_SHELF_BG_SAMPLE+offset_val,fgs,grouped=True,stacked=True)
#
#             offset_val += augs * NUM_SHELF_BG_SAMPLE
#
#     for id_class in range(len(cat_map)):
#         for iter in range(max_prod // min_prod):
#             batch = batches[id_class][iter]
#             augs = len(batch) // 2
#             for id in range(augs):
#                 fgs = batch[id * 2:(id + 1) * 2]
#                 with_neg_bg_bg_sampled(id * NUM_NEG_BG_SAMPLE + offset_val, fgs, stacked=True)
#
#             offset_val += augs * NUM_NEG_BG_SAMPLE



aug_annotation_file.close()
aug_annotation_file_with_cat.close()
aug_annotation_file_small_boxes.close()