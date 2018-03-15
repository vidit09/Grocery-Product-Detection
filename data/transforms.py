#Different image transformation using imgaug library

from imgaug import augmenters as iaa

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
