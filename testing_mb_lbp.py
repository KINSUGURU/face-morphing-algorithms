
# import cv2 as cv
# from extract_features import extract_feature_vector, crop_image

# # read an image
# image = cv.imread('raw/07026.png')

# cropped_image = crop_image(image=image)


# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])


# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])


# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])


# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])


# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])

# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])

# print(extract_feature_vector(image=cropped_image.copy(),plot=True)[:10])
from skimage import data
from matplotlib import pyplot as plt
from skimage.feature import draw_multiblock_lbp
from skimage.feature import multiblock_lbp
import numpy as np

for i in range(10):
    img = data.coins()

    mblbp_image = np.zeros(img.shape)

    for idx, x in np.ndenumerate(img):

        mblbp_image[idx[0],idx[1]] = multiblock_lbp(img, idx[0], idx[1], 3, 3)

    hist, bin = np.histogram(mblbp_image.ravel(),256,[0,255])

    print(hist[:10])