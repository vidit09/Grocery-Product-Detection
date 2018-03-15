import cv2
import numpy as np
from imgaug import augmenters as iaa
import os,glob
import os.path as osp

import yaml


from helper import resize
from transforms import aug_transforms
from warp import wrap_fg_bg

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir,'augmentation_param.yml'), 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


NUM_SHELF_BG_SAMPLE = cfg['NUM_SHELF_BG_SAMPLE']
NUM_NEG_BG_SAMPLE   = cfg['NUM_NEG_BG_SAMPLE']
TARGET_SIZE         = cfg['TARGET_SIZE']
MAX_SIZE            = cfg['MAX_SIZE']

out_dir             = cfg['OUT_DIR']




if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_img_dir = out_dir+'Images/'
if not os.path.exists(out_img_dir):
    os.mkdir(out_img_dir)

product_img_dir = cfg['PRODUCT_IMG_DIR']
product_img_path = product_img_dir+'TrainingFiles.txt'
product_cat_exp = 'Training/Food'

cat_map = []
with open(osp.join(cur_dir,'cat_mapping.txt')) as f:
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



bg_dir = cfg['BG_DIR']
bg_annotation = bg_dir + 'Annotations/'
bg_annotation = glob.glob(bg_annotation+'*.txt')

bg_annotation_grouped = bg_dir + 'Annotations_Grouped/'
bg_annotation_grouped = glob.glob(bg_annotation_grouped+'*.txt')

bg_img = bg_dir + 'Images_all/'
bg_img = glob.glob(bg_img+'*.jpg')

neg_bg_path = cfg['NEG_BG_DIR']
neg_bg_img = glob.glob(neg_bg_path+'*.jpg')


aug_annotation_file = open(out_dir+'annotations.txt','w')
aug_annotation_file_with_cat = open(out_dir+'annotations_with_big_box_cat.txt','w')
aug_annotation_file_small_boxes = open(out_dir+'annotations_with_small_box_cat.txt','w')



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

        im_scale, im_bg = resize(im_bg, TARGET_SIZE, MAX_SIZE)
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