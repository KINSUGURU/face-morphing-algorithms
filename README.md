# Face Morphing Algorithms

This reposiotry is a part of my research regrarding Face Morphing. This includes links, python notebooks and scripts used for experimental Face Morphing Generation and Detection. I also used open source code. Check the references and sources for details.

## Details

### What is face morphing


## Usage

### Requirements
It is advised to use a [Python virtual enviroment](https://docs.python.org/3/library/venv.html) or [conda/miniconda](https://docs.conda.io/projects/miniconda/en/latest/). The version of Python used is ```3.11.2```. Code last tested on Debian 12.

```
$ git clone https://github.com/mataktelis11/face-morphing-algorithms.git
$ python3 -m venv morphenv
$ source morphenv/bin/activate
$ pip install -r requirements.txt
```
Note: the ```image_aligner.py``` uses an older verion of the Pillow library (9.5.0)

### Example usage


for more information see the following

### Image alignment

Use ```image_aligner.py```
```
$ python image_aligner.py -h

usage: image_aligner.py [-h] -r RAW -a ALIGNED

options:
  -h, --help            show this help message and exit

Required arguments:
  -r RAW, --raw RAW     Provide the folder path containing the raw images.
  -a ALIGNED, --aligned ALIGNED
                        Provide the folder path for the results (aligned images).
```


### Morphing Generation

There are several known techniques for face morphing generation:

- Landmark based: OpenCV, Facemorpher and more
- GAN based: StyleGAN2, MIPGAN2
- Diffusion based: using interpolation with Diffusion autoencoders (this is a relatively new approach)

This repository has the notebook ```stylegan2_face_morphing_basics.ipynb``` as an experimental introduction to StyleGAN2 face morphing. Open with Google colab.

We also provide the script ```opencv_morph.py```. As the title suggests it generates morphs with OpenCV.

```
$ python opencv_morph.py -h

usage: opencv_morph.py [-h] [-a ALPHA] -s SRC -m MORPHED -p PAIRS

options:
  -h, --help            show this help message and exit
  -a ALPHA, --alpha ALPHA
                        Provide the morphing's alpha value [0, 1] (default: 0.5)

Required arguments:
  -s SRC, --src SRC     Provide the folder path containing the raw images.
  -m MORPHED, --morphed MORPHED
                        Provide the folder path for the results.
  -p PAIRS, --pairs PAIRS
                        Provide the file path of the `.csv` file containing the names of the pair of images to be morphed.
```

### Morphing Detection



## References and sources