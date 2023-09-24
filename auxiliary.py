
import argparse
import os
import bz2
import wget

def download_dlib_lmd(download_path):
    dlib_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    if not os.path.exists(download_path):
        
        print('Downloading dlib face landmarks detector...')
        
        tmp_file = wget.download(dlib_url)

        print("")

        with bz2.BZ2File(tmp_file, 'rb') as src, open(download_path, 'wb') as dst:
            dst.write(src.read())
        print("Success !")
    else:
        print(f'dlib landmark detector already downloaded in {download_path}')




def check_float_range(arg, MIN_VAL=0.0, MAX_VAL=1.0):
    '''Type function for argparse - a float within the predefined bounds.'''
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number.")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f

def check_dir_path(path):
    '''Checks if the given folder path as an argument exists.'''
    if os.path.isdir(path) or path == 'results':
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path.")

def check_dir_file(path):
    '''Checks if the given file path as an argument exists.'''
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file.")
