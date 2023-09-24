
import matplotlib.pyplot as plt

import os
import cv2 as cv
from skimage.feature import multiblock_lbp
import numpy as np
import csv
import argparse
import auxiliary as aux
from tqdm import tqdm

def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='extract_features.py',
                    description='A CLI tool for extracting texture feature from images of faces.')

    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize cropping and feature extraction (only when processing a single image)')  # on/off flag
    parser.add_argument('-i', '--image', type=aux.check_dir_file, help='Provide the path of a single image file for feature extraction.')
    parser.add_argument('-f', '--folder', type=aux.check_dir_path, help='Provide the folder path containing images for feature extraction.')

    requiredArgs = parser.add_argument_group('Required arguments')
    
    requiredArgs.add_argument('-o', '--output', help='Name of file that will contain feature vector(s).', required=True)

    return parser.parse_args()



def crop_image(image, plot=False):

    # crop images with hard coded points (assumed that images were aligned with image_aligner.py)
    point_1x = 300
    point_1y = 342

    point_2x = 717
    point_2y = 856

    # visualize cropping
    if(plot):

        plt.figure(figsize=(7,7))

        plt.subplot(1,2,1)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        cropped = image[point_1y:point_2y, point_1x:point_2x]

        plt.subplot(1,2,2)
        plt.imshow(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))

        plt.show()

        pass
    
    return image[point_1y:point_2y, point_1x:point_2x]




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



def main():

    # Parse arguments
    args = parse_arguments()


    if args.image and args.folder:
        raise argparse.ArgumentTypeError("Cannot use --image and --folder together")

    if not(args.image) and not(args.folder):
        raise argparse.ArgumentTypeError("You must specify a folder or an image as input")
    
    if args.image:
        print(f"Processing image {args.image}")

        image = cv.imread(args.image)

        cropped_image = crop_image(image, args.visualize)

        v = extract_feature_vector(cropped_image, args.visualize)

        print(v)

        with open(args.output, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(v)

        print(f'Writen vector to file {args.output}')
    
    if args.folder:

        print(f"Processing images in {args.folder}")

        # load images with opencv
        images = [ cv.imread(os.path.join(args.folder, file)) for file in os.listdir(args.folder) ]

        cropped_images = [ crop_image(img) for img in images]

        pbar = tqdm(total=len(cropped_images))

        with open(args.output, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for img in cropped_images:

                writer.writerow(extract_feature_vector(img))
                pbar.update(1)

        pbar.close()

        print(f'Writen vectors to file {args.output}')


if __name__ == "__main__":
    main()

