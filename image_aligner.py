
# source: https://github.com/woctezuma/stylegan2/tree/tiled-projector

import numpy as np
import scipy.ndimage
import os
import PIL.Image
import auxiliary as aux
import sys
import argparse

from LandmarksDetector import LandmarksDetector

DLIB_LMD_PATH = "shape_predictor_68_face_landmarks.dat"


def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='image_aligner.py',
                    description='A CLI tool for aligning images of faces using FFHQ\'s script.',
                    epilog='Disclaimer: this code comes from this reposioty: https://github.com/woctezuma/stylegan2/tree/tiled-projector')

    
    requiredArgs = parser.add_argument_group('Required arguments')

    requiredArgs.add_argument('-r', '--raw', type=aux.check_dir_path, help='Provide the folder path containing the raw images.', required=True)
    requiredArgs.add_argument('-a', '--aligned', type=aux.check_dir_path, help='Provide the folder path for the results (aligned images).', required=True)
    
    return parser.parse_args()

def create_aligned_image(src_file, out_file, face_landmarks,  output_size=1024, transform_size=4096, enable_padding=True):

        # Parse landmarks.
        # pylint: disable=unused-variable
        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1

        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load unaligned image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS) # ANTIALIAS is deprecated and will be removed in Pillow 10
                                                                              # you have to downgrade your pillow installation
                                                                              # pip install Pillow==9.5.0
                                                                              # source: https://stackoverflow.com/questions/76616042/attributeerror-module-pil-image-has-no-attribute-antialias

        # Save aligned image.
        img.save(out_file,'png')



def main():
    # Parse arguments
    args = parse_arguments()

    # download dlib model
    aux.download_dlib_lmd(DLIB_LMD_PATH)


    landmarksDetector = LandmarksDetector(DLIB_LMD_PATH)

    for img_name in os.listdir(args.raw):

        raw_img_path = os.path.join(args.raw, img_name)

        if raw_img_path.endswith('.ppm'):
            # https://github.com/dhar174/ppm-2-png/blob/master/ppm-2-png.py
            
            temp_image = PIL.Image.open(raw_img_path)
            new_path = raw_img_path[:-4]
            new_path = new_path + ".png"

            temp_image.save(new_path)

            print(f'Saved .ppm image as .png')
            print(f' {raw_img_path} \t -> \t {new_path}')
     
            raw_img_path = new_path

        for i, face_landmarks in enumerate(landmarksDetector.get_landmarks(raw_img_path), start=1):

            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(args.aligned, face_img_name)
            os.makedirs(args.aligned, exist_ok=True)
            create_aligned_image(raw_img_path, aligned_face_path, face_landmarks)


if __name__ == "__main__":
    main()
