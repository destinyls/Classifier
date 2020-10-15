import cv2

import torch
import numpy as np

from albumentations import *

def mixup_data(x, y, alpha=1.0, use_cuda=True):
 
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def letterbox_image(img, expected_size):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
    amat = np.dot(tmat, smat)
    amat = amat[:2, :]
    dst = cv2.warpAffine(img, amat, expected_size)
    return dst


def scale_image(image, expected_size):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_AREA)
    return image

def crop_image(image):
    if image.shape[1] / image.shape[0] > 0.66:
        image = CenterCrop(height=image.shape[0], width=int(0.66 * image.shape[0]), p=1)(image=image)['image']
    else:
        image = Crop(x_min=0, y_min=0, x_max=image.shape[1], y_max=image.shape[0], p=1.0)(image=image)['image']
    return image

