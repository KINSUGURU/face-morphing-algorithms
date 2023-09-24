
import matplotlib.pyplot as plt

import os
import cv2 as cv
from skimage.feature import multiblock_lbp
from skimage.feature import local_binary_pattern
import numpy as np
import csv

images = [cv.imread("raw/" + file) for file in os.listdir("raw/")]
images_morphed = [cv.imread("morphed/" + file) for file in os.listdir("morphed/")]

# crop images with hard coded points (assumed that images were aligned before morph)
point_1x = 300
point_1y = 342

point_2x = 717
point_2y = 856


# visualize cropping

# plt.figure(figsize=(7,7))

# for i,img in enumerate(images):

#     #img = cv.rectangle(img, (point_1x, point_1y), (point_2x, point_2y), (0, 255, 0), 3)

#     plt.subplot(2,len(images),i+1)
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

#     cropped = img[point_1y:point_2y, point_1x:point_2x]

#     plt.subplot(2,len(images),i+1+len(images))
#     plt.imshow(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))


# plt.figure(figsize=(7,7))

# for i,img in enumerate(images_morphed):

#     #img = cv.rectangle(img, (point_1x, point_1y), (point_2x, point_2y), (0, 255, 0), 3)

#     plt.subplot(2,len(images_morphed),i+1)
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

#     cropped = img[point_1y:point_2y, point_1x:point_2x]

#     plt.subplot(2,len(images_morphed),i+1+len(images_morphed))
#     plt.imshow(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))

# plt.show()


cropped_images = [ img[point_1y:point_2y, point_1x:point_2x] for img in images]

cropped_images_morphed = [ img[point_1y:point_2y, point_1x:point_2x] for img in images_morphed]

# plt.figure(figsize=(7,7))
# for i,img in enumerate(cropped_images):

#     plt.subplot(1,len(cropped_images),i+1)
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


# plt.figure(figsize=(7,7))
# for i,img in enumerate(cropped_images_morphed):

#     plt.subplot(1,len(cropped_images_morphed),i+1)
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# plt.show()



def extract_feature_vector(image, plot=False):
  
    # resize
    resized = cv.resize(image, (300,300), interpolation = cv.INTER_AREA)

    # convert to gray scale
    resized = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)

    mblbp_image = np.zeros((300,300))

    for idx, x in np.ndenumerate(resized):

        mblbp_image[idx[0],idx[1]] = multiblock_lbp(resized, idx[0], idx[1], 3, 3)

    hist, bin = np.histogram(mblbp_image.ravel(),256,[0,255])


    if(plot):

        plt.figure(figsize=(17,7))

        plt.subplot(1,3,1).set_title("Starting Image")
        plt.imshow(resized, cmap='gray', vmin=0, vmax=255)

        plt.subplot(1,3,2).set_title("MB-LBP image")
        plt.imshow(mblbp_image, cmap='gray', vmin=0, vmax=255)

        plt.subplot(1,3,3).set_title("MB-LBP histogram")
        plt.xlim([0,255])
        plt.plot(hist)

        plt.show()


    return hist

# print("hist")
# print(type(hist))
# print(hist.shape)

# print(type(bin))
# print(bin.shape)


# df = pd.DataFrame([hist], dtype = int)


# print(df)

filename = 'vecs1.csv'

with open(filename, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)


    for img in cropped_images:

        writer.writerow(extract_feature_vector(img))

filename = 'vecs2.csv'

with open(filename, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)


    for img in cropped_images_morphed:

        writer.writerow(extract_feature_vector(img))

