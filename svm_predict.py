import argparse
import cv2 as cv
from extract_features import crop_image, extract_feature_vector
from auxiliary import check_dir_file, check_dir_path
import joblib
import os
from tqdm import tqdm
import pandas as pd

def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='extract_features.py',
                    description='A CLI tool for detecting morphed images with a trained SVM model')

    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize cropping and feature extraction (only when processing a single image)')  # on/off flag
    parser.add_argument('-a', '--all', action='store_true', help='Display result for every single file')  # on/off flag
    parser.add_argument('-i', '--image', type=check_dir_file, help='Provide the path of a single image file for morphing detection.')
    parser.add_argument('-c', '--csv', type=check_dir_file, help='Provide the path of a single .csv file containig feature vectors for the SVM.')
    parser.add_argument('-f', '--folder', type=check_dir_path, help='Provide the folder path containing images for morphing detection.')

    requiredArgs = parser.add_argument_group('Required arguments')

    requiredArgs.add_argument('-m', '--model', type=check_dir_file, help='Provide the file path of the trained model savefile.', required=True)

    return parser.parse_args()





def main():

    # Parse arguments
    args = parse_arguments()

    has_image = args.image != None
    has_folder = args.folder != None
    has_csv = args.csv != None


    if sum([has_image, has_folder, has_csv]) > 1:
        raise argparse.ArgumentTypeError("Cannot use --image, --csv or --folder together. Use only one")

    if sum([has_image, has_folder, has_csv]) == 0:
        raise argparse.ArgumentTypeError("You must specify a folder or an image or a csv as input!")
    
    if args.image:
        print(f"Processing image {args.image}")

        image = cv.imread(args.image)

        cropped_image = crop_image(image, args.visualize)

        v = extract_feature_vector(cropped_image, args.visualize)

        # load the model from disk
        loaded_model = joblib.load(args.model)

        predict = loaded_model.predict(v.reshape(1, -1))

        label = ['bona fide', 'morphed']

        print(f'Result: class = {predict} = {label[int(predict[0])]}')
    
    if args.folder:

        print(f"Processing images in {args.folder}")

        label = ['bona fide', 'morphed']

        results = [0, 0]

        
        if not args.all:
            pbar = tqdm(total=len(os.listdir(args.folder)))

        for file in os.listdir(args.folder):

            image = cv.imread(os.path.join(args.folder, file)) 
        
            cropped_image = crop_image(image)

            v = extract_feature_vector(cropped_image)

            # load the model from disk
            loaded_model = joblib.load(args.model)

            #print(v)

            predict = loaded_model.predict(v.reshape(1, -1))

            if args.all:
                print(f'Result for image {file}: class = {predict} = {label[int(predict[0])]}')

            if predict[0] == 0:
                results[0] += 1
            else:
                results[1] += 1

            if not args.all:
                pbar.update(1)

        if not args.all:
            pbar.close()

        print("Result report:")
        print(f" \t {results[0]} images classified as 'bona fide'")
        print(f" \t {results[1]} images classified as 'morphed'")


    if args.csv:

        print(f"Processing images in {args.csv}")

        label = ['bona fide', 'morphed']

        df = pd.read_csv(filepath_or_buffer=args.csv, header=None)

        #print(df)

        # load the model from disk
        loaded_model = joblib.load(args.model)

        predict = loaded_model.predict(df)

        #print(predict)

        print("Result report:")
        print(f" \t {len([x for x in predict if x==0])} images classified as 'bona fide'")
        print(f" \t {len([x for x in predict if x==1])} images classified as 'morphed'")


if __name__ == "__main__":
    main()

