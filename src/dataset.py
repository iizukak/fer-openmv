"""
Python dataset loader for FER2013 dataset

Refercne:
> Challenges in Representation Learning: Facial Expression Recognition Challenge
> https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

FER2013 file structure is here

fer2013
├── fer2013.bib
├── fer2013.csv
└── README

fer2013.csv is the main file.
This csv file contains tree columns: emotion,pixels,Usage

Each column means
- emotion: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
- pixels: 2304 values in [0, 255]. This is the gray scale value for 48 * 48
- Training, PrivateTest, PublicTest
"""

import csv
import numpy as np

IMAGE_SIZE = 48


def _load_csv(path):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        lines = [row for row in reader]
    return lines[1:]  # pass the header


def _parse(lines):
    """
    parse raw FER2013 data to numpy array

    return
    (x_train, y_train), (x_public_test, y_public_test), (x_private_test, y_private_test)

    data format: (n, 48, 48) as NHW
    label format: (1,)
    """
    train = [line for line in lines if line[2] == "Training"]
    public_test = [line for line in lines if line[2] == "PublicTest"]
    private_test = [line for line in lines if line[2] == "PrivateTest"]

    train_num = len(train)
    public_test_num = len(public_test)
    private_test_num = len(private_test)

    assert train_num + public_test_num + private_test_num == len(lines)

    # Initialize Variables with empty ndarrays
    x_train = np.zeros((train_num, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    y_train = np.zeros(train_num, dtype=np.uint8)

    x_public_test = np.zeros((public_test_num, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    y_public_test = np.zeros(public_test_num, dtype=np.uint8)

    x_private_test = np.zeros((private_test_num, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    y_private_test = np.zeros(private_test_num, dtype=np.uint8)

    for i in range(train_num):
        x_train[i] = np.array(train[i][1].split(' '), np.float32).reshape((IMAGE_SIZE, IMAGE_SIZE))
        y_train[i] = train[i][0]

    for i in range(public_test_num):
        x_public_test[i] = np.array(public_test[i][1].split(' '), np.float32).reshape((IMAGE_SIZE, IMAGE_SIZE))
        y_public_test[i] = public_test[i][0]

    for i in range(private_test_num):
        x_private_test[i] = np.array(private_test[i][1].split(' '), np.float32).reshape((IMAGE_SIZE, IMAGE_SIZE))
        y_private_test[i] = private_test[i][0]

    return (x_train, y_train), (x_public_test, y_public_test), (x_private_test, y_private_test)


def load_dataset(path):
    return _parse(_load_csv(path))
