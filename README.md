# MPEG7 Descriptor Toolkit

## Description
This repository presents a toolkit for working with MPEG7 descriptors in image processing tasks. It provides functionalities for training, evaluating, and querying image descriptors using Python and OpenCV.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Descriptors](#descriptors)
- [Dataset](#dataset)
- [Results](#results)

## Overview
This project offers a set of scripts for working with MPEG7 descriptors, including training descriptor models on a dataset, evaluating their performance, and querying image with this dataset.

## Installation
To use this toolkit, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
The toolkit provides three main functionalities: training, evaluation, and querying of MPEG7 descriptors.

### Training
To train a descriptor model, use the following command:
```bash
python main.py train [directory] [descriptor] [output_file]
```

* `directory`: Path to the directory containing training images.
* `descriptor`: Descriptor method to use. Choose from available options: CLD, DCD, or CSD.
* `output_file`: Path to the output file to save the trained model.

During training, the toolkit computes descriptor vectors for images in the training dataset using the specified method and saves the results to a text file.

![image](https://github.com/lucas-gtr/MPEG7_Descriptors/assets/12534925/7b3815ba-0881-4ece-8fac-701dd6dd6486)

### Evaluation
To evaluate the performance of a descriptor model, execute the command:
```bash
python main.py eval [directory] [descriptor] [descriptor_database]
```

* `directory`: Path to the directory containing test images.
* `descriptor`: Descriptor method to use. Choose from available options: CLD, DCD, or CSD.
* `output_file`: Path to the descriptor database file.

The toolkit evaluates the performance of a descriptor method on a test dataset by computing the mean average precision (mAP) of the predictions.

![image](https://github.com/lucas-gtr/MPEG7_Descriptors/assets/12534925/4917cb8a-749b-485f-95e3-aadc56a332b7)

### Query
To query a descriptor model on a particular image, execute the command:
```bash
python main.py query [query_image] [descriptor] [descriptor_database]
```

* `directory`: Path to the query image.
* `descriptor`: Descriptor method to use. Choose from available options: CLD, DCD, or CSD.
* `output_file`: Path to the descriptor database file.

For query tasks, the toolkit utilizes a trained descriptor model to find similar images in the descriptor database based on a query image.

![image](https://github.com/lucas-gtr/MPEG7_Descriptors/assets/12534925/90b4d1fe-fd0b-4124-9762-b23c5223fa7f)

## Descriptors
The toolkit supports three MPEG7 descriptor methods:

* **Color Layout Descriptor (CLD)**: Represents the spatial layout of colors in an image. It divides the image into a grid and computes color moments for each grid cell.

* **Dominant Color Descriptor (DCD)**: Captures the dominant colors in an image. It quantizes the color space and computes the frequency of each color cluster.

* **Color Structure Descriptor (CSD)**: Describes the spatial arrangement of color structures in an image. It analyzes the distribution of color pairs in predefined directions.

Each descriptor has its own parameters. They can be modified in the [config file](config.py).

## Dataset

To effectively utilize the descriptor, the dataset must follow this specific structure: it must be in a directory where subfolders are named in accordance with their respective labels, and each subfolder must contain relevant images corresponding to its label.

```
dataset/
│
├── label1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── label2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── ...
```

The image name are not relevant. The image extension allower are `.jpg` `.jpeg` `.bmp` and `.png`

## Results

To assess the performance of the descriptor models, we conducted experiments on two distinct datasets: a [fruit dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification) and a [flower dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). The results of these experiments are summarized in the following tables:

### Fruit Dataset Results

The fruits dataset is composed of 200 training images and 100 testing images. They are divided in 5 fruits : 
* Apple
* Banana
* Mango
* Grape
* Strawberry

| Descriptor Method | Training time | Descriptor size | Testing time | Mean Average Precision (mAP) |
|-------------------|---------------|-----------------|--------------|------------------------------|
| CLD               | 0.5s          | 12              | 0.6s         | 46.35%                       |
| DCD               | 16.6s         | 33              | 13.3s        | 36.99%                       |
| CSD               | 2.7s          | 64              | 2.0s         | 45.91%                       |

### Flower Dataset Results

The fruits dataset is composed of 3041 training images and 1276 testing images. They are divided in 5 flowers : 
* Daisy
* Dandelion
* Rose
* Tulip
* Sunflower

| Descriptor Method | Training time | Descriptor size | Testing time | Mean Average Precision (mAP) |
|-------------------|---------------|-----------------|--------------|------------------------------|
| CLD               | 12.7s         | 12              | 78.2s        | 51.61%                       |
| DCD               | 568.2s        | 33              | 1677.3s      | 43.12%                       |
| CSD               | 122.6s        | 64              | 171.7s       | 50.62%                       |

These tables provide insights into the performance of each descriptor method on the respective datasets. The results indicate that the Color Layout Descriptor (CLD) is the best over the other methods across both datasets, achieving the highest mean average precision for the lowest training time and smallest descriptors.

Looking carefully at the dataset, we notice that it is not very precise and many images don't fit well with their label (very distant images for example). Therefore, it would be more beneficial to evaluate these descriptors on a higher-quality dataset.
