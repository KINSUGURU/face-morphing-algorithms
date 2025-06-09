import numpy as np
import os
import PIL.Image
import auxiliary as aux
import sys
import argparse

from LandmarksDetector import LandmarksDetector

DLIB_LMD_PATH = "shape_predictor_68_face_landmarks.dat"

def parse_arguments():
    parser = argparse.ArgumentParser()

    requiredArgs = parser.add_argument_group('Required arguments')
    requiredArgs.add_argument('-r', '--raw', type=aux.check_dir_path, required=True, help='Path to raw images.')
    requiredArgs.add_argument('-a', '--aligned', type=aux.check_dir_path, required=True, help='Output root path (will contain aligned/ and aligned_landmark/)')

    return parser.parse_args()

def create_aligned_image(src_file, out_file, face_landmarks, output_size=256):
    """Resize image to output_size and scale landmarks accordingly."""
    img = PIL.Image.open(src_file).convert('RGB')
    w, h = img.size
    scale_x = output_size / w
    scale_y = output_size / h

    # Resize image
    img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
    img.save(out_file, 'png')

    # Scale landmarks
    lm = np.array(face_landmarks, dtype=np.float32)
    lm[:, 0] *= scale_x
    lm[:, 1] *= scale_y
    return lm

def main():
    args = parse_arguments()
    aux.download_dlib_lmd(DLIB_LMD_PATH)

    output_root = os.path.abspath(args.aligned)
    aligned_dir = os.path.join(output_root, "aligned")
    landmarks_dir = os.path.join(output_root, "aligned_landmark")

    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)

    fail_list_path = os.path.join(landmarks_dir, "fail_list.txt")
    with open(fail_list_path, 'w') as fail_file:
        detector = LandmarksDetector(DLIB_LMD_PATH)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')

        for img_name in os.listdir(args.raw):
            if not img_name.lower().endswith(valid_exts):
                continue

            raw_img_path = os.path.join(args.raw, img_name)

            if raw_img_path.endswith('.ppm'):
                temp_image = PIL.Image.open(raw_img_path)
                new_path = raw_img_path[:-4] + ".png"
                temp_image.save(new_path)
                print(f"Converted .ppm to .png: {raw_img_path} -> {new_path}")
                raw_img_path = new_path
                img_name = os.path.basename(new_path)

            landmarks_all = list(detector.get_landmarks(raw_img_path))

            if not landmarks_all:
                print(f"[WARN] No face detected: {img_name}")
                fail_file.write(img_name + "\n")
                continue

            for face_landmarks in landmarks_all:
                aligned_path = os.path.join(aligned_dir, img_name)
                landmark_path = os.path.join(landmarks_dir, os.path.splitext(img_name)[0] + ".txt")

                transformed_landmarks = create_aligned_image(
                    raw_img_path, aligned_path, face_landmarks,
                    output_size=256
                )

                if transformed_landmarks is not None:
                    with open(landmark_path, 'w') as f:
                        for (x, y) in transformed_landmarks:
                            f.write(f"{x:.6f} {y:.6f}\n")

if __name__ == "__main__":
    main()