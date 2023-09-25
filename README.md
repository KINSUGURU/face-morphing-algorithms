# Face Morphing Algorithms

This reposiotry is a part of my research regrarding Face Morphing. This includes links, python notebooks and scripts used for experimental Face Morphing Generation and Detection. The code is heavly based on existing open source code. Check the references and sources for details.


### What is face morphing

There are several known techniques for face morphing generation:

- Landmark based: OpenCV, Facemorpher and more
- GAN based: StyleGAN2, MIPGAN2
- Diffusion based: using interpolation with Diffusion autoencoders (this is a relatively new approach)

This repository has the notebook ```stylegan2_face_morphing_basics.ipynb``` as an experimental introduction to StyleGAN2 face morphing. Open with Google colab.

We also provide the script ```opencv_morph.py```. As the title suggests it generates morphs with OpenCV.


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

1. Align your images
```
$ python image_aligner.py -r raw/ -a aligned/
```
Use ```image_aligner.py``` as seen above to align your images. Input images are in folder ```raw``` and output images in folder ```aligned```. The code used comes from the official [FFHQ Dataset repository](https://github.com/NVlabs/ffhq-dataset) and is also implemented in [woctezuma's fork of stylegan2](https://github.com/woctezuma/stylegan2/tree/tiled-projector).

2. Generate Morphs
```
$ python opencv_morph.py -s aligned/ -m morphs/ -p ~pairs.csv
```
This is script is used to generate landmark-based OpenCV morphs. The code is from this [repository](https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate) and is based on [learnopencv.com](https://learnopencv.com/face-morph-using-opencv-cpp-python/).

3. Extract feature vectors

```
$ python extract_features.py -f aligned/ -o bona_fide_vectors.csv
$ python extract_features.py -f morphed/ -o morphed_vectors.csv
```
The feature vectors are extracted by using **Multi-Block Local Binary Patterns**. This is a texture based feaure extraction and it creates a new image from which we use the histogram as a feature vector.

4. Train SVM

Now we have our vectors from the two classes in separate .csv files. We can train our own model.

```
$ python svm_train.py -b bona_fide_vectors.csv -m morphed_vectors.csv -o my_model.sav -v
```

5. Use the model

```
$ python svm_predict.py  -m my_model.sav -i my_image.png -v
```

## References and sources

I suggest you check them out if you are currently researching for the topic of Face Morphing.

- [bob.paper.icassp2022_morph_generate](https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate)
- [learnopencv.com](https://learnopencv.com/face-morph-using-opencv-cpp-python/)
- [FFHQ Dataset repository](https://github.com/NVlabs/ffhq-dataset)
- [woctezuma's fork of stylegan2](https://github.com/woctezuma/stylegan2/tree/tiled-projector)
