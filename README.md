# Face Analysis

## Introduction
This project aims to analysis humans' faces from the real-time video stream.
1. Circle the faces appeared in the screen.
2. For each face, predict his/ her age and gender.

## Model
The weights of the well-trained age and gender detect models are listed below:

|  Model   | Weights| Paper |
|  ----  | ----  | ----  |
| Age Detect Model  | [model structure](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt), [model weights](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel)| [DEX: Deep EXpectation of apparent age from a single image](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)|
| Gender Detect Model  | [model structure](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt), [model weights](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel) | [DEX: Deep EXpectation of apparent age from a single image](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)|

## Requirements

### 1. Dependencies

The dependencies are listed in `requirements.txt`. For quick deployment:
```
python -m pip install -r requirements.txt
```

### 2. Models

The weights of the well-trained age and gender detect models can be accessed via the link provided in the last section. Please downloads all of the four files and put them in the `model/` folder. For quick deployment:
```
bash download.sh
```

### 3. Folder Structure
Finally, the folder structure should looks like:
```
|-- root
    |-- face_detect.py
    |-- README.md
    |-- requirements.txt
    |-- model
    |   |-- age.prototxt
    |   |-- dex_chalearn_iccv2015.caffemodel
    |   |-- gender.prototxt
    |   |-- gender.caffemodel
    |-- download.sh
```

## Quick Start

**NOTES** :
1. Your laptop or computer should have a camera.
2. Please give this program authority to use the camera.

```
python face_detect.py
```

## Reference

[1] Rothe, Rasmus, Radu Timofte, and Luc Van Gool. "Dex: Deep expectation of apparent age from a single image." Proceedings of the IEEE international conference on computer vision workshops. 2015.

[2] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.
