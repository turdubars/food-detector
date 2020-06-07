## Food-Detector
![Python 3.8](https://img.shields.io/badge/python-v3.8-blue?style=flat)
![MXNet](https://img.shields.io/badge/mxnet-v1.6-blue?style=flat)
![Docker](https://img.shields.io/badge/docker-v19.03-blue?style=flat)
![MIT License](https://img.shields.io/github/license/turdubars/food-detector?style=flat&color=green)

## Installation
### Docker
If you don't want to deal with packages, install with Docker Compose
```bash
docker-compose up --build
```
Run shell in the container with
```bash
docker-compose run food-detector bash
```

### Pip
If you want to install without docker, just run
```bash
pip install -r requirements.txt
```

## Usage

To predict food in the image file
```bash
python food-detector --file <path to image file>
```
To predict food in the image with images's url
```bash
python food-detector --url <url of image>
```
Run --help to get the additional information
```bash
python food-detector --help
```
