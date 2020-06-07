## Food Detector
![Python 3.8](https://img.shields.io/badge/python-v3.8-blue?style=flat)
![MXNet](https://img.shields.io/badge/mxnet-v1.6-blue?style=flat)
![Docker](https://img.shields.io/badge/docker-v19.03-blue?style=flat)
![MIT License](https://img.shields.io/github/license/turdubars/food-detector?style=flat&color=green)

![Food Detector Demo](readme/demo.png)


A custom model to detect local food using two CNN models YOLOv3 and custom ResNet-50.
YOLOv3 model was pretrained on COCO Dataset and ResNet-50 was pretrained on Imagenet and finetuned for the custom dataset of local food that was manually collected from Google Images with Python and Javascript.
![Model structure](readme/structure.png)
## Installation
### Docker
If you don't want to deal with packages, install with Docker Compose
```bash
docker-compose up --build
```
And run shell in the container with
```bash
docker-compose run food-detector bash
```

### Pip
If you want to install with pip instead of Docker
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with a local image file
```bash
python food-detector.py --file <path to image file>
```
For example:
```bash
python food-detector.py --file test_images/test0.jpg
```
**Note that during the first run the application automatically will download YOLOv3 parameters**

By default the application saves images with prediction in 'predictions' folder

To predict food in an image from internet use -u or --url flag
```bash
python food-detector.py --url <url of image>
```
There are other flags to print outputs (-p), to save predicted images (-w), to set a threshold (-t)\
Run -h or --help to get the additional information
```bash
python food-detector.py --help
```
