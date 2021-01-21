import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from .utils import load_image, equalize
from pydicom import dcmread
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
BUFFER_SIZE = 32
BATCH_SIZE = 8
img_height, img_width = 512, 512


def get_data(target_size=512, test_size=0.25, random_state=23,preprocessing=True, label_type='region',file_csv="data/public-annotations.csv",data_directory="data/public-cohen-subset"):
    """
    Get train and test data from cohen dataset
    :param target_size: image dimension
    :param test_size: train test split (default 0.25)
    :param random_state: random seed
    :param preprocessing: whether to apply preprocessing
    :param label: gt label ['senior', 'junior']
    :param label_type:
    :return: 3x2 region score or global one ['global', 'region']
    """

    assert label_type in ['global', 'region'], print("label_type must be either 'global' or 'region'.")

    # load annotations from csv
    ds = pd.read_csv(file_csv,sep=";",dtype=str)

    X = []
    y = []
    for it in tqdm(ds.itertuples()):

        im = dcmread( f"{data_directory}/{it.Filename}").pixel_array
        
        im = np.array(im)
        brixia_txt = str(it.BrixiaScore)
        brixia = np.array([int(b) for b in str(brixia_txt)])
        bs = np.reshape(brixia,(2,3)).T
        if label_type == 'global':
            bs = np.sum(bs)

        X.append(im)
        y.append(bs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def get_data_iterator(target_size=512, test_size=0.25, random_state=23,preprocessing=True, label_type='region',file_csv="data/public-annotations.csv",data_directory="data/public-cohen-subset"):
    assert label_type in ['global', 'region'], print("label_type must be either 'global' or 'region'.")
    # load annotations from csv
    ds = pd.read_csv(file_csv,sep=";",dtype=str)
    X = []
    y = []
    for it in tqdm(ds.itertuples()):
        imageFile = f"{data_directory}/{it.Filename}"
        brixia_txt = str(it.BrixiaScore)
        brixia = np.array([int(b) for b in str(brixia_txt)])
        bs = np.reshape(brixia,(2,3)).T
        X.append(imageFile)
        y.append(bs)
    
    XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size=test_size, random_state=random_state)
    

    DSTrain = tf.data.Dataset.from_tensor_slices((XTrain,YTrain))
    DSTest = tf.data.Dataset.from_tensor_slices((XTest,YTest))
    def decode_img(img_path):
        image_bytes = tf.io.read_file(img_path)
        img = tfio.image.decode_dicom_image(image_bytes)
        img = (img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))
        img = equalize(img)*256
        return tf.image.resize(img, [img_height, img_width])


    DSTrain.shuffle(buffer_size=BUFFER_SIZE)
    DSTest.shuffle(buffer_size=BUFFER_SIZE)
    DSTrain = DSTrain.map(lambda x_,y_: (decode_img(x_)[0,:,:],y_))
    DSTest = DSTest.map(lambda x_,y_: (decode_img(x_)[0,:,:],y_))
    DSTrain = DSTrain.batch(BATCH_SIZE)
    DSTest = DSTest.batch(BATCH_SIZE)

    
    return DSTrain, DSTest
    