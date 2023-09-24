
import os
import bz2

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


