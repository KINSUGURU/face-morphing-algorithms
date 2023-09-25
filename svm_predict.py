import argparse
import cv2 as cv
from extract_features import crop_image, extract_feature_vector
from auxiliary import check_dir_file, check_dir_path
import joblib

def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='extract_features.py',
                    description='A CLI tool for detecting morphed images with a trained SVM model')

    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize cropping and feature extraction (only when processing a single image)')  # on/off flag
    parser.add_argument('-a', '--all', action='store_true', help='Display result for every single file')  # on/off flag
    parser.add_argument('-i', '--image', type=check_dir_file, help='Provide the path of a single image file for morphing detection.')
    parser.add_argument('-f', '--folder', type=check_dir_path, help='Provide the folder path containing images for morphing detection.')

    requiredArgs = parser.add_argument_group('Required arguments')

    requiredArgs.add_argument('-m', '--model', type=check_dir_file, help='Provide the file path of the trained model savefile.', required=True)

    return parser.parse_args()





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

        # load the model from disk
        loaded_model = joblib.load('finalized_model.sav')


        predict = loaded_model.predict(v.reshape(1, -1))

        label = ['bona fide', 'morphed']

        print(f'Result: class = {predict} = {label[int(predict)]}')
    
    if args.folder:

        print(f"Processing images in {args.folder}")





if __name__ == "__main__":
    main()

