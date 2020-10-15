import os
import cv2

import numpy as np
from PIL import Image
from skimage import transform as trans
from skimage.io import imshow, imread, imsave

from albumentations import *

image_path = "./dataset/train/calling_images"

def get_3rd_point(point_a, point_b):
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c

def get_transfrom_matrix(center_scale, output_size):
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32)

def affine_image(img, width, height):
    center = np.array([i / 2 for i in img.size], dtype=np.float32)
    size = np.array([i for i in img.size], dtype=np.float32)

    center_size = [center, size]
    trans_affine = get_transfrom_matrix(
        center_size,
        [width, height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    img = img.transform(
        (width, height),
        method=Image.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.BILINEAR,
    )
    return img


# Crop(x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0)(image=image)['image']

if __name__ == "__main__":
    for image_name in os.listdir(image_path):
        image_name = os.path.join(image_path, image_name)
        image = imread(image_name)
        print(image.shape)

        if image.shape[1] / image.shape[0] > 0.66:
            image = CenterCrop(height=image.shape[0], width=int(0.66 * image.shape[0]), p=1)(image=image)['image']
        else:
            image = Crop(x_min=0, y_min=0, x_max=image.shape[1], y_max=image.shape[0], p=1.0)(image=image)['image']

        cv2.imshow("Image", image)
        cv2.waitKey(500)
    print("Hello World")